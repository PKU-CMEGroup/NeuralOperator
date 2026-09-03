"""Frozen PCNO data/model primitives for the SU2 Unsteady NACA0012 R0 test.

The module deliberately separates three concerns:

* fail-closed validation of the frozen experiment contract and its parent
  trajectory/phase evidence;
* deterministic conversion of native SU2 states and the fixed VTK9 mesh into
  model-ready arrays; and
* the complete-BDF2 residual PCNO used by the clean baseline.

It does not train a model, open sealed roles, or evaluate a scientific result.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from pcno.geo_utility import compute_edge_gradient_weights
from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.su2_naca_phase_pilot import (
    NACA_PHASE_PILOT_ANALYSIS_SCHEMA,
    load_verified_phase_pilot_receipt,
)
from utility.time_dependent_no.su2_naca_trajectory import (
    NACA_TRAJECTORY_RECEIPT_FILENAME,
    NACA_TRAJECTORY_STORAGE_FILENAME,
    NACA_TRAJECTORY_STORAGE_SCHEMA,
    load_verified_trajectory_receipt,
)
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
    lumped_vertex_areas,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    NACA_RESTART_FIELDS,
    RESOURCE_MANIFEST_SCHEMA,
    SU2Mesh,
    SU2Restart,
    parse_su2_mesh,
    read_su2_binary_restart,
    sha256_file,
)

BASELINE_CONTRACT_SCHEMA = "time_dependent_no.su2_naca0012_pcno_baseline.v1"
# Updated only when the human-owned frozen contract is intentionally revised.
BASELINE_CONTRACT_SHA256 = (
    "94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87"
)
NUM_DYNAMIC_FIELDS = 5
NUM_STATIC_FEATURES = 6
NUM_INPUT_FEATURES = 16
NACA_BOUND_MESH_SHA256 = (
    "20dfc96875b1d1a8377c2e64248b6ee42a1ecf525c50d822fb5f539d06d68d15"
)
NACA_BOUND_MESH_NUM_POINTS = 14576
NACA_BOUND_MESH_NUM_ELEMENTS = 14336
NACA_INPUT_FEATURE_NAMES = (
    "normalized_coordinate_x",
    "normalized_coordinate_y",
    "quadrature_density",
    "interior_one_hot",
    "airfoil_one_hot",
    "farfield_one_hot",
    *(f"normalized_previous_{field}" for field in NACA_DYNAMIC_FIELDS),
    *(f"normalized_current_{field}" for field in NACA_DYNAMIC_FIELDS),
)
ROLE_ORDER = ("train", "development", "prospective", "sealed")
OPEN_ROLES = frozenset(("train", "development"))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIED_CONTRACT_TOKEN = object()
_VERIFIED_CONTEXT_TOKEN = object()


class VerifiedNACAContract(Mapping[str, Any]):
    """Immutable handle for the one exact baseline-contract byte identity."""

    __slots__ = ("__file_sha256", "__payload_json", "__payload_sha256")

    def __init__(
        self,
        payload: Mapping[str, Any],
        file_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_CONTRACT_TOKEN:
            raise TypeError("verified contracts may only be created by the loader")
        payload_json = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        )
        object.__setattr__(self, "_VerifiedNACAContract__payload_json", payload_json)
        object.__setattr__(
            self,
            "_VerifiedNACAContract__payload_sha256",
            _canonical_payload_sha256(payload),
        )
        object.__setattr__(self, "_VerifiedNACAContract__file_sha256", str(file_sha256))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified contracts are immutable")

    def _payload(self, token: object) -> dict[str, Any]:
        if token is not _VERIFIED_CONTRACT_TOKEN:
            raise PermissionError("verified contract payload is internal")
        payload = json.loads(self.__payload_json)
        if _canonical_payload_sha256(payload) != self.__payload_sha256:
            raise ValueError("verified contract payload identity changed")
        return payload

    @property
    def file_sha256(self) -> str:
        return self.__file_sha256

    def to_mapping(self) -> dict[str, Any]:
        """Return an untrusted copy suitable only for serialization."""

        return json.loads(self.__payload_json)

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self):
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())


class NACAVerifiedContext:
    """Opaque verified parent packet with bound storage, mesh, and diagnostics."""

    __slots__ = (
        "__contract",
        "__diagnostic_records_json",
        "__mesh_contract_json",
        "__mesh_record_json",
        "__repository",
        "__resource_manifest_sha256",
        "__resource_root",
        "__storage_json",
        "__trajectory_root",
    )

    def __init__(
        self,
        *,
        contract: VerifiedNACAContract,
        repository: Path,
        trajectory_root: Path,
        resource_root: Path,
        resource_manifest_sha256: str,
        storage: Mapping[str, Any],
        mesh_record: Mapping[str, Any],
        mesh_contract: Mapping[str, Any],
        diagnostic_records: Mapping[str, Any],
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_CONTEXT_TOKEN:
            raise TypeError(
                "verified contexts may only be created by parent validation"
            )
        object.__setattr__(self, "_NACAVerifiedContext__contract", contract)
        object.__setattr__(self, "_NACAVerifiedContext__repository", repository)
        object.__setattr__(
            self, "_NACAVerifiedContext__trajectory_root", trajectory_root
        )
        object.__setattr__(self, "_NACAVerifiedContext__resource_root", resource_root)
        object.__setattr__(
            self,
            "_NACAVerifiedContext__resource_manifest_sha256",
            _require_digest(resource_manifest_sha256, "Stage-0 resource manifest"),
        )
        for name, value in (
            ("storage_json", storage),
            ("mesh_record_json", mesh_record),
            ("mesh_contract_json", mesh_contract),
            ("diagnostic_records_json", diagnostic_records),
        ):
            object.__setattr__(
                self,
                f"_NACAVerifiedContext__{name}",
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified contexts are immutable")

    @property
    def contract(self) -> VerifiedNACAContract:
        return self.__contract

    @property
    def repository_root(self) -> Path:
        return self.__repository

    @property
    def trajectory_root(self) -> Path:
        return self.__trajectory_root

    @property
    def resource_root(self) -> Path:
        return self.__resource_root

    @property
    def resource_manifest_sha256(self) -> str:
        return self.__resource_manifest_sha256

    @property
    def mesh_path(self) -> Path:
        record = json.loads(self.__mesh_record_json)
        return _safe_child(self.__resource_root, record["file"], "mesh")

    @property
    def mesh_record(self) -> dict[str, Any]:
        return json.loads(self.__mesh_record_json)

    def _payloads(
        self, token: object
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if token is not _VERIFIED_CONTEXT_TOKEN:
            raise PermissionError("verified context payloads are internal")
        return (
            self.__contract._payload(_VERIFIED_CONTRACT_TOKEN),
            json.loads(self.__storage_json),
            json.loads(self.__mesh_contract_json),
            json.loads(self.__diagnostic_records_json),
        )


def _require_regular_file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file is absent or aliased: {path}")
    return path


def _resolve_real_directory(path: str | Path, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"{label} is aliased")
    if not supplied.is_dir():
        raise ValueError(f"{label} is absent")
    return supplied.resolve()


def _read_with_stable_file_identity(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
    reader: Callable[[Path], Any],
) -> Any:
    """Read one file only while its exact identity remains unchanged."""

    file_path = _require_regular_file(path)
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise TypeError(f"expected {label} byte count must be an integer")
    digest = _require_digest(expected_sha256, label)
    before = (file_path.stat().st_size, sha256_file(file_path))
    if before[0] != expected_bytes:
        raise ValueError(f"{label} byte count differs")
    if before[1] != digest:
        raise ValueError(f"{label} SHA256 differs")
    result = reader(file_path)
    _require_regular_file(file_path)
    after = (file_path.stat().st_size, sha256_file(file_path))
    if after != before:
        raise ValueError(f"{label} changed while it was being read")
    return result


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _load_self_hashed_json(path: Path, schema: str) -> dict[str, Any]:
    _require_regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported schema")
    observed = payload.pop("canonical_payload_sha256", None)
    expected = _canonical_payload_sha256(payload)
    if observed != expected:
        raise ValueError(f"{path}: canonical payload SHA256 differs")
    payload["canonical_payload_sha256"] = observed
    return payload


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _inclusive_range(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must be a two-integer inclusive range")
    first, last = value
    if first > last:
        raise ValueError(f"{label} is reversed")
    return first, last


def _strict_indices(values: Iterable[Any], label: str) -> tuple[int, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError(f"{label} must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError(f"{label} must contain only integer indices")
    converted = tuple(int(value) for value in raw)
    if len(set(converted)) != len(converted):
        raise ValueError(f"{label} must not contain duplicates")
    return converted


def _validate_population_payload(contract: Mapping[str, Any]) -> None:
    """Validate all frozen role blocks, anchors, gaps, and leakage bounds."""

    population = contract.get("phase_population")
    if not isinstance(population, Mapping):
        raise TypeError("contract phase_population must be an object")
    period = float(population.get("measured_period_steps", math.nan))
    horizon = population.get("primary_rollout_horizon_steps")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("measured period must be positive and finite")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("primary rollout horizon must be a positive integer")
    if horizon != round(6.0 * period):
        raise ValueError("primary rollout horizon does not implement the frozen rule")
    periods = float(population.get("primary_rollout_horizon_periods", math.nan))
    if not math.isclose(periods, horizon / period, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reported primary horizon periods are inconsistent")
    if population.get("reported_horizons_steps") != [1, 35, 104, 208]:
        raise ValueError("reported rollout horizons differ from the frozen contract")
    if population.get("primary_long_horizon_gate_steps") != horizon:
        raise ValueError("long-horizon gate differs from the primary horizon")
    if population.get("post_outcome_extension_forbidden") is not True:
        raise ValueError("post-outcome horizon extension must remain forbidden")

    roles = population.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != ROLE_ORDER:
        raise ValueError("population roles must appear in the frozen order")
    owned_ranges: list[tuple[int, int, str]] = []
    all_k: list[int] = []
    for role_index, role in enumerate(ROLE_ORDER):
        record = roles.get(role)
        if not isinstance(record, Mapping):
            raise TypeError(f"population role {role} must be an object")
        owned_first, owned_last = _inclusive_range(
            record.get("owned_frame_indices_inclusive"),
            f"{role} owned frames",
        )
        owned_ranges.append((owned_first, owned_last, role))
        k_values = record.get("k_values")
        anchors = record.get("anchor_input_indices")
        if not isinstance(k_values, list) or not isinstance(anchors, list):
            raise TypeError(f"{role} k values and anchors must be lists")
        if len(k_values) != 8 or len(anchors) != 8:
            raise ValueError(f"{role} must contain exactly eight phase anchors")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*k_values, *anchors)
        ):
            raise ValueError(f"{role} k values and anchors must be integers")
        if any(value % 4 != role_index for value in k_values):
            raise ValueError(f"{role} k values violate k modulo four assignment")
        if anchors != sorted(set(anchors)):
            raise ValueError(f"{role} anchors must be strictly increasing")
        for anchor in anchors:
            if anchor - 1 < owned_first or anchor + horizon > owned_last:
                raise ValueError(f"{role} anchor {anchor} violates rollout leakage")
        all_k.extend(k_values)

        dense = record.get("dense_transition_center_indices_inclusive")
        if role in OPEN_ROLES:
            dense_first, dense_last = _inclusive_range(
                dense, f"{role} dense transitions"
            )
            if dense_first - 1 < owned_first or dense_last + 1 > owned_last:
                raise ValueError(f"{role} dense transitions violate BDF2 leakage")
            expected_count = dense_last - dense_first + 1
            if record.get("dense_transition_count") != expected_count:
                raise ValueError(f"{role} dense transition count differs")
        elif dense is not None:
            raise ValueError(
                f"sealed role {role} may not declare open dense transitions"
            )

    if sorted(all_k) != list(range(32)):
        raise ValueError("phase k values must partition 0 through 31")
    for (_, previous_last, previous_role), (next_first, _, next_role) in pairwise(
        owned_ranges
    ):
        if previous_last >= next_first:
            raise ValueError(
                f"owned frame blocks overlap: {previous_role} and {next_role}"
            )

    gaps = population.get("unused_gap_indices_inclusive")
    if not isinstance(gaps, list):
        raise TypeError("unused gaps must be a list")
    observed_gaps = [
        _inclusive_range(gap, f"unused gap {index}") for index, gap in enumerate(gaps)
    ]
    expected_internal = [
        (left[1] + 1, right[0] - 1)
        for left, right in pairwise(owned_ranges)
        if left[1] + 1 <= right[0] - 1
    ]
    trailing = (
        owned_ranges[-1][1] + 1,
        int(contract["parent_evidence"]["trajectory"]["last_output_index"]),
    )
    if trailing[0] <= trailing[1]:
        expected_internal.append(trailing)
    if observed_gaps != expected_internal:
        raise ValueError("unused gaps do not exactly separate the role blocks")


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != BASELINE_CONTRACT_SCHEMA:
        raise ValueError("unsupported NACA PCNO baseline contract schema")
    if (
        contract.get("status")
        != "frozen_before_population_materialization_or_pcno_outcome"
    ):
        raise ValueError("baseline contract is not in the frozen pre-outcome state")
    state = contract.get("state_and_dataset")
    if not isinstance(state, Mapping):
        raise TypeError("state_and_dataset must be an object")
    if state.get("dynamic_fields") != list(NACA_DYNAMIC_FIELDS):
        raise ValueError("dynamic fields differ from the native five-field contract")
    if (
        state.get("native_state_dtype") != "float64"
        or state.get("stored_model_state_dtype") != "float64"
    ):
        raise ValueError("native and stored model states must remain float64")
    geometry = state.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry contract must be an object")
    if (
        geometry.get("mesh") != "fixed native SU2 quadrilateral mesh"
        or geometry.get("adjacency") != "element"
        or geometry.get("gradient_rcond") != 0.001
        or geometry.get("spectral_measure")
        != (
            "one physical vertex-lumped area measure computed only by "
            "utility.time_dependent_no.su2_native_replay.lumped_vertex_areas, "
            "whose ordered-quad cell areas use the qualified polygon shoelace "
            "rule; auxiliary PCNO node_weights are m_i / sum_i m_i"
        )
    ):
        raise ValueError("geometry semantics differ from the frozen PCNO contract")
    expected_static = [
        "normalized coordinate x",
        "normalized coordinate y",
        "quadrature density",
        "interior one-hot",
        "airfoil one-hot",
        "farfield one-hot",
    ]
    if geometry.get("static_input_features") != expected_static:
        raise ValueError("static input feature order differs")

    normalization = contract.get("normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("normalization contract must be an object")
    if (
        normalization.get("fit_frames") != [955, 1194]
        or normalization.get("fit_transitions") != [956, 1193]
        or normalization.get("weights") != "uniform node"
        or normalization.get("residual_mean_subtracted") is not False
        or normalization.get("development_or_sealed_data_may_change_normalization")
        is not False
    ):
        raise ValueError("normalization scope differs from the train-only contract")

    model = contract.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("model contract must be an object")
    expected_model = {
        "backbone": "pcno.PCNO",
        "history": "complete BDF2 pair",
        "input_channels": NUM_INPUT_FEATURES,
        "output_channels": NUM_DYNAMIC_FIELDS,
        "fourier_modes": [8, 8],
        "nmeasures": 1,
        "layers": [128, 128, 128, 128, 128],
        "projection_width": 128,
        "activation": "gelu",
        "gradient_branch": True,
        "zero_initialize_output_head": True,
        "zero_initialization": (
            "set both backbone.fc2.weight and backbone.fc2.bias exactly to zero "
            "before the first optimizer step"
        ),
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model contract field {key} differs")
    if len(NACA_INPUT_FEATURE_NAMES) != NUM_INPUT_FEATURES:
        raise AssertionError("internal NACA input layout is inconsistent")

    parent = contract.get("parent_evidence")
    if not isinstance(parent, Mapping):
        raise TypeError("parent_evidence must be an object")
    for section in ("phase_contract", "phase_analysis", "trajectory"):
        if not isinstance(parent.get(section), Mapping):
            raise TypeError(f"parent evidence section {section} must be an object")
    for label, value in (
        ("phase contract", parent["phase_contract"].get("sha256")),
        ("phase analysis file", parent["phase_analysis"].get("file_sha256")),
        (
            "phase analysis payload",
            parent["phase_analysis"].get("canonical_payload_sha256"),
        ),
        (
            "trajectory receipt file",
            parent["trajectory"].get("trajectory_receipt_file_sha256"),
        ),
        (
            "trajectory receipt payload",
            parent["trajectory"].get("trajectory_receipt_payload_sha256"),
        ),
        (
            "storage manifest file",
            parent["trajectory"].get("storage_manifest_file_sha256"),
        ),
        (
            "storage manifest payload",
            parent["trajectory"].get("storage_manifest_payload_sha256"),
        ),
        (
            "ordered restart records",
            parent["trajectory"].get("ordered_restart_records_sha256"),
        ),
        ("history", parent["trajectory"].get("history_sha256")),
    ):
        _require_digest(value, label)
    claims = contract.get("claim_boundary")
    if not isinstance(claims, Mapping) or any(
        value is not False for value in claims.values()
    ):
        raise ValueError("all pre-execution claim-boundary flags must remain false")
    _validate_population_payload(contract)


def _verified_contract_payload(
    contract: VerifiedNACAContract,
) -> dict[str, Any]:
    if not isinstance(contract, VerifiedNACAContract):
        raise TypeError("an exact VerifiedNACAContract is required")
    if contract.file_sha256 != BASELINE_CONTRACT_SHA256:
        raise ValueError("verified contract file identity differs")
    payload = contract._payload(_VERIFIED_CONTRACT_TOKEN)
    _validate_contract_semantics(payload)
    return payload


def validate_population_contract(contract: VerifiedNACAContract) -> None:
    """Revalidate the exact immutable baseline population contract."""

    _validate_population_payload(_verified_contract_payload(contract))


def load_naca_baseline_contract(path: str | Path) -> VerifiedNACAContract:
    """Load the exact frozen contract and validate its internal semantics."""

    contract_path = _require_regular_file(Path(path))
    contract_bytes = contract_path.read_bytes()
    observed_sha256 = sha256(contract_bytes).hexdigest()
    if observed_sha256 != BASELINE_CONTRACT_SHA256:
        raise ValueError("NACA PCNO baseline contract differs from the frozen SHA256")
    payload = json.loads(contract_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("NACA PCNO baseline contract must be a JSON object")
    _validate_contract_semantics(payload)
    return VerifiedNACAContract(
        payload,
        observed_sha256,
        _token=_VERIFIED_CONTRACT_TOKEN,
    )


def _load_verified_storage_manifest(path: str | Path) -> dict[str, Any]:
    """Load the trajectory storage manifest with schema and self-hash checks."""

    return _load_self_hashed_json(Path(path), NACA_TRAJECTORY_STORAGE_SCHEMA)


def _safe_child(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {label} path")
    supplied = root / path
    current = supplied
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{label} path is aliased")
        parent = current.parent
        if parent == current:
            raise ValueError(f"{label} does not descend from its declared root")
        current = parent
    candidate = supplied.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} escapes its declared root")
    return candidate


def validate_parent_evidence(
    contract: VerifiedNACAContract,
    *,
    repository_root: str | Path,
    resource_root: str | Path,
) -> NACAVerifiedContext:
    """Verify the phase decision and trajectory receipts bound by the contract.

    This validates the small provenance packet.  Individual native restart
    files are independently checked when they are extracted for a dataset.
    """

    contract_payload = _verified_contract_payload(contract)
    repository = _resolve_real_directory(repository_root, "repository root")
    resources_root = _resolve_real_directory(resource_root, "Stage-0 resource root")
    parent = contract_payload["parent_evidence"]

    phase_contract_record = parent["phase_contract"]
    phase_contract_path = _safe_child(
        repository / "docs" / "time_dependent_no",
        str(phase_contract_record["file"]),
        "phase contract",
    )
    _require_regular_file(phase_contract_path)
    if sha256_file(phase_contract_path) != phase_contract_record["sha256"]:
        raise ValueError("phase-pilot contract differs from its parent binding")

    phase_record = parent["phase_analysis"]
    phase_path = _safe_child(
        repository,
        str(phase_record["relative_artifact"]),
        "phase analysis",
    )
    _require_regular_file(phase_path)
    if sha256_file(phase_path) != phase_record["file_sha256"]:
        raise ValueError("phase analysis file differs from its parent binding")
    phase = load_verified_phase_pilot_receipt(phase_path)
    if phase.get("schema") != NACA_PHASE_PILOT_ANALYSIS_SCHEMA:
        raise ValueError("phase analysis schema differs")
    if (
        phase.get("outcome") != phase_record["outcome"]
        or phase.get("outcome") != "PHASE_ONLY_SUPPORTED"
        or phase.get("canonical_payload_sha256")
        != phase_record["canonical_payload_sha256"]
        or phase.get("error") is not None
    ):
        raise ValueError("phase analysis does not support the frozen population")
    if any(
        phase.get(key) is not False
        for key in (
            "phase_population_claimed",
            "pcno_training_claimed",
            "sealed_access_claimed",
        )
    ):
        raise ValueError("phase analysis exceeds its allowed claim boundary")

    trajectory_record = parent["trajectory"]
    trajectory_root = phase_path.parent
    receipt_path = trajectory_root / NACA_TRAJECTORY_RECEIPT_FILENAME
    storage_path = trajectory_root / NACA_TRAJECTORY_STORAGE_FILENAME
    for path, expected, label in (
        (
            receipt_path,
            trajectory_record["trajectory_receipt_file_sha256"],
            "trajectory receipt",
        ),
        (
            storage_path,
            trajectory_record["storage_manifest_file_sha256"],
            "storage manifest",
        ),
    ):
        _require_regular_file(path)
        if sha256_file(path) != expected:
            raise ValueError(f"{label} differs from the contract binding")
    receipt = load_verified_trajectory_receipt(receipt_path)
    storage = _load_verified_storage_manifest(storage_path)
    if (
        receipt.get("canonical_payload_sha256")
        != trajectory_record["trajectory_receipt_payload_sha256"]
        or storage.get("canonical_payload_sha256")
        != trajectory_record["storage_manifest_payload_sha256"]
    ):
        raise ValueError("trajectory packet payload binding differs")
    if (
        receipt.get("status") != "execution_and_validation_succeeded"
        or receipt.get("trajectory_validated") is not True
        or receipt.get("execution_succeeded") is not True
        or storage.get("status") != "validated"
    ):
        raise ValueError("trajectory parent is not successfully validated")

    output_contract = storage.get("output_contract")
    if not isinstance(output_contract, Mapping):
        raise TypeError("storage manifest lacks its output contract")
    expected_output = {
        "first_index": trajectory_record["first_output_index"],
        "last_index": trajectory_record["last_output_index"],
        "count": trajectory_record["output_count"],
        "native_fields": list(NACA_NATIVE_RESTART_FIELDS),
        "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
    }
    if dict(output_contract) != expected_output:
        raise ValueError("storage output contract differs from the baseline binding")
    aggregate = storage.get("aggregate")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("ordered_file_records_sha256")
        != trajectory_record["ordered_restart_records_sha256"]
    ):
        raise ValueError("ordered restart aggregate differs from the baseline binding")
    history = storage.get("history")
    if (
        not isinstance(history, Mapping)
        or history.get("sha256") != trajectory_record["history_sha256"]
    ):
        raise ValueError("trajectory history differs from the baseline binding")
    files = storage.get("files")
    if not isinstance(files, list) or len(files) != trajectory_record["output_count"]:
        raise ValueError("storage manifest has the wrong restart record count")

    packet = phase.get("packet")
    if not isinstance(packet, Mapping):
        raise TypeError("phase analysis lacks its verified packet binding")
    packet_checks = {
        "first_output_index": trajectory_record["first_output_index"],
        "last_output_index": trajectory_record["last_output_index"],
        "output_count": trajectory_record["output_count"],
        "ordered_restart_records_sha256": trajectory_record[
            "ordered_restart_records_sha256"
        ],
        "trajectory_receipt_payload_sha256": trajectory_record[
            "trajectory_receipt_payload_sha256"
        ],
        "storage_manifest_payload_sha256": trajectory_record[
            "storage_manifest_payload_sha256"
        ],
    }
    for key, expected in packet_checks.items():
        if packet.get(key) != expected:
            raise ValueError(f"phase packet field {key} differs")

    stage0_binding = packet.get("stage0_authority_binding")
    if not isinstance(stage0_binding, Mapping):
        raise TypeError("phase packet lacks its Stage-0 resource binding")
    if receipt.get("stage0_authority_binding") != stage0_binding:
        raise ValueError(
            "trajectory receipt Stage-0 binding differs from phase evidence"
        )
    manifest_name = stage0_binding.get("manifest_file")
    if manifest_name != "R0_SU2_NACA_RESOURCE_MANIFEST.json":
        raise ValueError("Stage-0 resource manifest filename differs")
    manifest_path = _safe_child(
        repository / "docs" / "time_dependent_no",
        manifest_name,
        "Stage-0 resource manifest",
    )
    _require_regular_file(manifest_path)
    manifest_sha256 = _require_digest(
        stage0_binding.get("manifest_sha256"), "Stage-0 resource manifest"
    )
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Stage-0 resource manifest differs from phase evidence")
    resource_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(resource_manifest, Mapping)
        or resource_manifest.get("schema") != RESOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError("Stage-0 resource manifest schema differs")
    resources = resource_manifest.get("resources")
    mesh_contract = resource_manifest.get("mesh_contract")
    resource_digests = stage0_binding.get("resource_sha256")
    if not isinstance(resources, Mapping) or not isinstance(mesh_contract, Mapping):
        raise TypeError("Stage-0 resource manifest lacks mesh evidence")
    if not isinstance(resource_digests, Mapping):
        raise TypeError("Stage-0 resource digest binding is absent")

    mesh_record = resources.get("mesh")
    packet_mesh = packet.get("mesh")
    if not isinstance(mesh_record, Mapping) or not isinstance(packet_mesh, Mapping):
        raise TypeError("verified phase evidence lacks a mesh record")
    exact_mesh_record = {
        key: mesh_record.get(key) for key in ("file", "bytes", "sha256")
    }
    if exact_mesh_record != dict(packet_mesh):
        raise ValueError("phase and resource-manifest mesh records differ")
    if mesh_record.get("sha256") != resource_digests.get("mesh"):
        raise ValueError("mesh digest differs from the Stage-0 authority binding")
    if mesh_record.get("sha256") != NACA_BOUND_MESH_SHA256:
        raise ValueError("mesh digest differs from the frozen NACA mesh identity")
    if (
        mesh_contract.get("num_points") != NACA_BOUND_MESH_NUM_POINTS
        or mesh_contract.get("num_elements") != NACA_BOUND_MESH_NUM_ELEMENTS
    ):
        raise ValueError("mesh cardinality differs from the frozen NACA identity")
    mesh_path = _safe_child(resources_root, str(mesh_record["file"]), "mesh")
    _require_regular_file(mesh_path)
    if mesh_path.stat().st_size != int(mesh_record.get("bytes", -1)):
        raise ValueError("bound live mesh byte count differs")
    if sha256_file(mesh_path) != _require_digest(mesh_record.get("sha256"), "mesh"):
        raise ValueError("bound live mesh SHA256 differs")

    diagnostic_records: dict[str, dict[str, Any]] = {}
    for index in (497, 498, 499):
        key = f"restart_{index:05d}"
        record = resources.get(key)
        if not isinstance(record, Mapping):
            raise TypeError(f"Stage-0 resource manifest lacks {key}")
        expected_name = f"restart_flow_{index:05d}.dat"
        exact_record = {
            field: record.get(field) for field in ("file", "bytes", "sha256")
        }
        if exact_record["file"] != expected_name:
            raise ValueError(f"diagnostic record {index} filename differs")
        if exact_record["sha256"] != resource_digests.get(key):
            raise ValueError(f"diagnostic record {index} digest binding differs")
        _require_digest(exact_record["sha256"], f"diagnostic record {index}")
        if isinstance(exact_record["bytes"], bool) or not isinstance(
            exact_record["bytes"], int
        ):
            raise TypeError(f"diagnostic record {index} byte count differs")
        diagnostic_path = _safe_child(
            resources_root, expected_name, f"diagnostic record {index}"
        )
        if not diagnostic_path.is_file() or diagnostic_path.is_symlink():
            raise ValueError(f"diagnostic record {index} file is absent or aliased")
        if diagnostic_path.stat().st_size != exact_record["bytes"]:
            raise ValueError(f"diagnostic record {index} byte count changed")
        if sha256_file(diagnostic_path) != exact_record["sha256"]:
            raise ValueError(f"diagnostic record {index} SHA256 changed")
        diagnostic_records[str(index)] = exact_record

    return NACAVerifiedContext(
        contract=contract,
        repository=repository,
        trajectory_root=trajectory_root,
        resource_root=resources_root,
        resource_manifest_sha256=manifest_sha256,
        storage=storage,
        mesh_record=exact_mesh_record,
        mesh_contract=mesh_contract,
        diagnostic_records=diagnostic_records,
        _token=_VERIFIED_CONTEXT_TOKEN,
    )


def _restart_record(storage: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    files = storage.get("files")
    if not isinstance(files, list):
        raise TypeError("storage manifest files must be a list")
    matches = [record for record in files if record.get("index") == index]
    if len(matches) != 1:
        raise ValueError(
            f"storage manifest has {len(matches)} records for index {index}"
        )
    record = matches[0]
    if not isinstance(record, Mapping):
        raise TypeError("restart file record must be an object")
    return record


def _open_role_record(contract: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    if role not in ROLE_ORDER:
        raise ValueError(f"unknown population role: {role}")
    if role not in OPEN_ROLES:
        raise PermissionError(f"population role {role} remains sealed")
    record = contract["phase_population"]["roles"][role]
    if not isinstance(record, Mapping):
        raise TypeError(f"population role {role} is invalid")
    return record


def _validate_role_frame(contract: Mapping[str, Any], role: str, index: Any) -> int:
    record = _open_role_record(contract, role)
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError("trajectory frame index must be an integer")
    frame_index = int(index)
    first, last = _inclusive_range(
        record["owned_frame_indices_inclusive"], f"{role} owned frames"
    )
    if frame_index < first or frame_index > last:
        raise ValueError(f"frame {frame_index} is outside the {role} owned block")
    return frame_index


def _coerce_restart(restart: str | Path | SU2Restart) -> SU2Restart:
    return (
        restart
        if isinstance(restart, SU2Restart)
        else read_su2_binary_restart(_require_regular_file(Path(restart)))
    )


def _verify_and_extract_dynamic_state(
    parsed: SU2Restart,
    *,
    expected_sha256: str | None,
    expected_num_points: int | None,
    expected_coordinates: np.ndarray | None,
) -> np.ndarray:
    if expected_sha256 is not None and parsed.sha256 != _require_digest(
        expected_sha256, "expected restart"
    ):
        raise ValueError("restart SHA256 differs from its storage record")
    if expected_num_points is not None and parsed.num_points != int(
        expected_num_points
    ):
        raise ValueError("restart point count differs from the fixed mesh")
    coordinate_columns = [parsed.fields.index("x"), parsed.fields.index("y")]
    coordinates = parsed.values[:, coordinate_columns]
    if expected_coordinates is not None:
        expected = np.asarray(expected_coordinates, dtype=np.float64)
        if expected.shape != coordinates.shape or not np.array_equal(
            coordinates, expected
        ):
            raise ValueError("restart coordinates differ from the fixed native mesh")
    dynamic_columns = [parsed.fields.index(field) for field in NACA_DYNAMIC_FIELDS]
    state = np.array(parsed.values[:, dynamic_columns], dtype=np.float64, copy=True)
    if state.shape != (parsed.num_points, NUM_DYNAMIC_FIELDS) or not np.all(
        np.isfinite(state)
    ):
        raise ValueError("restart dynamic state is invalid")
    return state


def _extract_naca_state(
    restart: str | Path | SU2Restart,
    *,
    expected_sha256: str | None = None,
    expected_num_points: int | None = None,
    expected_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Extract five fields from an exact canonical-17 or native-19 restart."""

    parsed = _coerce_restart(restart)
    if parsed.fields not in (NACA_RESTART_FIELDS, NACA_NATIVE_RESTART_FIELDS):
        raise ValueError(
            "restart is neither the exact canonical-17 nor native-19 schema"
        )
    return _verify_and_extract_dynamic_state(
        parsed,
        expected_sha256=expected_sha256,
        expected_num_points=expected_num_points,
        expected_coordinates=expected_coordinates,
    )


def _extract_native_state(
    restart: str | Path | SU2Restart,
    *,
    expected_sha256: str | None = None,
    expected_num_points: int | None = None,
    expected_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Extract the five evolved fields from an exact native 19-field restart."""

    parsed = _coerce_restart(restart)
    if parsed.fields != NACA_NATIVE_RESTART_FIELDS or parsed.num_fields != 19:
        raise ValueError("restart does not have the exact native 19-field schema")
    return _verify_and_extract_dynamic_state(
        parsed,
        expected_sha256=expected_sha256,
        expected_num_points=expected_num_points,
        expected_coordinates=expected_coordinates,
    )


def extract_verified_trajectory_state(
    context: NACAVerifiedContext,
    role: str,
    index: int,
    *,
    expected_coordinates: np.ndarray,
) -> np.ndarray:
    """Extract one verified frame, refusing protected roles before packet access."""

    if not isinstance(context, NACAVerifiedContext):
        raise TypeError("a NACAVerifiedContext is required")
    contract = _verified_contract_payload(context.contract)
    frame_index = _validate_role_frame(contract, role, index)
    _, storage, _, _ = context._payloads(_VERIFIED_CONTEXT_TOKEN)
    record = _restart_record(storage, frame_index)
    expected_name = f"trajectory_flow_{frame_index:05d}.dat"
    if record.get("file") != expected_name:
        raise ValueError("restart record filename differs from its index")
    root = context.trajectory_root
    if not root.is_dir() or root.is_symlink():
        raise ValueError("trajectory root is absent or aliased")
    path = _safe_child(root, expected_name, "restart")
    _require_regular_file(path)
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("restart byte count differs from its storage record")
    digest = _require_digest(record.get("sha256"), "restart record")
    if sha256_file(path) != digest:
        raise ValueError("restart SHA256 differs from its storage record")
    return _extract_native_state(
        path,
        expected_sha256=digest,
        expected_num_points=np.asarray(expected_coordinates).shape[0],
        expected_coordinates=expected_coordinates,
    )


def extract_verified_diagnostic_state(
    context: NACAVerifiedContext,
    index: int,
    *,
    expected_coordinates: np.ndarray,
) -> np.ndarray:
    """Extract only the bound diagnostic-only canonical 497/498/499 triplet."""

    if not isinstance(context, NACAVerifiedContext):
        raise TypeError("a NACAVerifiedContext is required")
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError("diagnostic frame index must be an integer")
    frame_index = int(index)
    if frame_index not in (497, 498, 499):
        raise PermissionError("only diagnostic frames 497/498/499 are accessible")
    _, _, _, records = context._payloads(_VERIFIED_CONTEXT_TOKEN)
    record = records.get(str(frame_index))
    if not isinstance(record, Mapping):
        raise TypeError(f"diagnostic frame {frame_index} is not bound")
    expected_name = f"restart_flow_{frame_index:05d}.dat"
    if record.get("file") != expected_name:
        raise ValueError("diagnostic restart filename differs from its index")
    root = context.resource_root
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Stage-0 resource root is absent or aliased")
    path = _safe_child(root, expected_name, "diagnostic restart")
    _require_regular_file(path)
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("diagnostic restart byte count differs")
    digest = _require_digest(record.get("sha256"), "diagnostic restart")
    if sha256_file(path) != digest:
        raise ValueError("diagnostic restart SHA256 differs")
    return _extract_naca_state(
        path,
        expected_sha256=digest,
        expected_num_points=np.asarray(expected_coordinates).shape[0],
        expected_coordinates=expected_coordinates,
    )


@dataclass(frozen=True)
class NACAGeometry:
    """Immutable fixed geometry arrays used by every NACA PCNO presentation."""

    native_coordinates: np.ndarray
    elements: np.ndarray
    node_measures: np.ndarray
    node_weights: np.ndarray
    quadrature_density: np.ndarray
    boundary_one_hot: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    fourier_lengths: np.ndarray
    gradient_rcond: float = 0.001

    def __post_init__(self) -> None:
        float_arrays = (
            "native_coordinates",
            "node_measures",
            "node_weights",
            "quadrature_density",
            "boundary_one_hot",
            "edge_gradient_weights",
            "fourier_lengths",
        )
        int_arrays = ("elements", "directed_edges")
        for name in float_arrays:
            value = np.array(
                getattr(self, name), dtype=np.float64, copy=True, order="C"
            )
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in int_arrays:
            value = np.array(getattr(self, name), dtype=np.int64, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gradient_rcond", float(self.gradient_rcond))
        self._validate()

    @property
    def num_nodes(self) -> int:
        return int(self.native_coordinates.shape[0])

    @property
    def translated_nodes(self) -> np.ndarray:
        return self.native_coordinates - np.min(self.native_coordinates, axis=0)

    @property
    def normalized_coordinates(self) -> np.ndarray:
        return self.translated_nodes / self.fourier_lengths

    @property
    def static_features(self) -> np.ndarray:
        return np.concatenate(
            (
                self.normalized_coordinates,
                self.quadrature_density,
                self.boundary_one_hot,
            ),
            axis=-1,
        )

    def _validate(self) -> None:
        nodes = self.native_coordinates
        if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.all(np.isfinite(nodes)):
            raise ValueError("native coordinates must be finite [N,2]")
        num_nodes = nodes.shape[0]
        if self.elements.ndim != 2 or self.elements.shape[1] != 5:
            raise ValueError("quad elements must have shape [E,5]")
        if not np.all(self.elements[:, 0] == 2):
            raise ValueError("every PCNO quad must use [2,n0,n1,n2,n3] encoding")
        if np.any(self.elements[:, 1:] < 0) or np.any(
            self.elements[:, 1:] >= num_nodes
        ):
            raise ValueError("quad elements reference an unknown node")
        for name in ("node_measures", "node_weights", "quadrature_density"):
            value = getattr(self, name)
            if value.shape != (num_nodes, 1) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [N,1]")
        if np.any(self.node_measures <= 0.0) or np.any(self.node_weights <= 0.0):
            raise ValueError("physical measures and weights must be positive")
        expected_weights = self.node_measures / np.sum(self.node_measures)
        if not np.allclose(self.node_weights, expected_weights, rtol=1e-13, atol=0.0):
            raise ValueError("node weights are not the normalized physical measure")
        expected_density = self.node_weights / self.node_measures
        if not np.allclose(
            self.quadrature_density, expected_density, rtol=1e-13, atol=0.0
        ):
            raise ValueError(
                "quadrature density differs from weight divided by measure"
            )
        if self.boundary_one_hot.shape != (num_nodes, 3):
            raise ValueError("boundary one-hot must have shape [N,3]")
        if not np.all((self.boundary_one_hot == 0.0) | (self.boundary_one_hot == 1.0)):
            raise ValueError("boundary one-hot must contain only zero and one")
        if not np.all(np.sum(self.boundary_one_hot, axis=1) == 1.0):
            raise ValueError("every node must have exactly one boundary class")
        if self.directed_edges.ndim != 2 or self.directed_edges.shape[1] != 2:
            raise ValueError("directed edges must have shape [D,2]")
        if self.edge_gradient_weights.shape != self.directed_edges.shape:
            raise ValueError(
                "edge gradient weights must align with two-dimensional edges"
            )
        if np.any(self.directed_edges < 0) or np.any(self.directed_edges >= num_nodes):
            raise ValueError("directed edges reference an unknown node")
        if np.any(self.directed_edges[:, 0] == self.directed_edges[:, 1]):
            raise ValueError("directed graph may not contain self edges")
        if not np.all(np.isfinite(self.edge_gradient_weights)):
            raise ValueError("edge gradient weights must be finite")
        if self.fourier_lengths.shape != (2,) or np.any(self.fourier_lengths <= 0.0):
            raise ValueError("Fourier lengths must contain two positive extents")
        expected_lengths = np.ptp(nodes, axis=0)
        if not np.array_equal(self.fourier_lengths, expected_lengths):
            raise ValueError(
                "Fourier lengths differ from exact native coordinate extents"
            )
        if not math.isfinite(self.gradient_rcond) or self.gradient_rcond != 0.001:
            raise ValueError("gradient rcond differs from the frozen value")
        if self.static_features.shape != (num_nodes, NUM_STATIC_FEATURES):
            raise AssertionError("static feature layout is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        """Return a plain, NPZ-compatible copy of all persistent arrays."""

        return {
            "geometry_schema": np.asarray("time_dependent_no.naca_geometry.v1"),
            "native_coordinates": np.array(self.native_coordinates, copy=True),
            "elements": np.array(self.elements, copy=True),
            "node_measures": np.array(self.node_measures, copy=True),
            "node_weights": np.array(self.node_weights, copy=True),
            "quadrature_density": np.array(self.quadrature_density, copy=True),
            "boundary_one_hot": np.array(self.boundary_one_hot, copy=True),
            "directed_edges": np.array(self.directed_edges, copy=True),
            "edge_gradient_weights": np.array(self.edge_gradient_weights, copy=True),
            "fourier_lengths": np.array(self.fourier_lengths, copy=True),
            "gradient_rcond": np.asarray(self.gradient_rcond, dtype=np.float64),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NACAGeometry:
        schema = np.asarray(value.get("geometry_schema")).item()
        if schema != "time_dependent_no.naca_geometry.v1":
            raise ValueError("unsupported NACA geometry schema")
        return cls(
            native_coordinates=value["native_coordinates"],
            elements=value["elements"],
            node_measures=value["node_measures"],
            node_weights=value["node_weights"],
            quadrature_density=value["quadrature_density"],
            boundary_one_hot=value["boundary_one_hot"],
            directed_edges=value["directed_edges"],
            edge_gradient_weights=value["edge_gradient_weights"],
            fourier_lengths=value["fourier_lengths"],
            gradient_rcond=float(np.asarray(value["gradient_rcond"]).item()),
        )

    def expand(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Create stride-zero batch views of the immutable fixed geometry."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("geometry floating dtype must be floating point")
        target = torch.device(device)

        def floating(value: np.ndarray) -> torch.Tensor:
            base = torch.as_tensor(
                np.array(value, copy=True), dtype=dtype, device=target
            )
            return base.unsqueeze(0).expand(batch_size, *base.shape)

        def integer(value: np.ndarray) -> torch.Tensor:
            base = torch.as_tensor(
                np.array(value, copy=True), dtype=torch.int64, device=target
            )
            return base.unsqueeze(0).expand(batch_size, *base.shape)

        node_mask = (
            torch.ones((self.num_nodes, 1), dtype=dtype, device=target)
            .unsqueeze(0)
            .expand(batch_size, self.num_nodes, 1)
        )
        return {
            "static_features": floating(self.static_features),
            "nodes": floating(self.translated_nodes),
            "native_coordinates": floating(self.native_coordinates),
            "node_measures": floating(self.node_measures),
            "node_weights": floating(self.node_weights),
            "node_mask": node_mask,
            "boundary_one_hot": floating(self.boundary_one_hot),
            "directed_edges": integer(self.directed_edges),
            "edge_gradient_weights": floating(self.edge_gradient_weights),
        }


class VerifiedNACAGeometry:
    """Immutable geometry whose mesh and serialized artifact were verified."""

    __slots__ = ("__geometry", "__source_mesh_sha256")

    def __init__(
        self,
        geometry: NACAGeometry,
        source_mesh_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_CONTEXT_TOKEN:
            raise TypeError("verified geometry requires an internal provenance check")
        if not isinstance(geometry, NACAGeometry):
            raise TypeError("verified geometry must wrap NACAGeometry")
        if source_mesh_sha256 != NACA_BOUND_MESH_SHA256:
            raise ValueError("verified geometry source mesh differs")
        if (
            geometry.num_nodes != NACA_BOUND_MESH_NUM_POINTS
            or geometry.elements.shape[0] != NACA_BOUND_MESH_NUM_ELEMENTS
        ):
            raise ValueError("verified geometry cardinality differs")
        object.__setattr__(self, "_VerifiedNACAGeometry__geometry", geometry)
        object.__setattr__(
            self,
            "_VerifiedNACAGeometry__source_mesh_sha256",
            source_mesh_sha256,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified geometry is immutable")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__geometry, name)

    @property
    def source_mesh_sha256(self) -> str:
        return self.__source_mesh_sha256

    def to_mapping(self) -> dict[str, Any]:
        mapping = self.__geometry.to_mapping()
        mapping.update(
            {
                "verified_geometry_schema": np.asarray(
                    "time_dependent_no.naca_verified_geometry.v1"
                ),
                "baseline_contract_sha256": np.asarray(BASELINE_CONTRACT_SHA256),
                "source_mesh_sha256": np.asarray(self.__source_mesh_sha256),
            }
        )
        return mapping

    def expand(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        return self.__geometry.expand(batch_size, device, dtype=dtype)

    def _geometry(self, token: object) -> NACAGeometry:
        if token is not _VERIFIED_CONTEXT_TOKEN:
            raise PermissionError("verified geometry payload is internal")
        return self.__geometry


def load_verified_naca_geometry(
    contract: VerifiedNACAContract,
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> VerifiedNACAGeometry:
    """Load a provenance-marked NPZ after verifying its packet file record."""

    _verified_contract_payload(contract)
    geometry_path = Path(path)

    def read_archive(file_path: Path) -> dict[str, Any]:
        with np.load(file_path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    mapping = _read_with_stable_file_identity(
        geometry_path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label="serialized geometry",
        reader=read_archive,
    )
    if (
        np.asarray(mapping.get("verified_geometry_schema")).item()
        != "time_dependent_no.naca_verified_geometry.v1"
        or np.asarray(mapping.get("baseline_contract_sha256")).item()
        != BASELINE_CONTRACT_SHA256
        or np.asarray(mapping.get("source_mesh_sha256")).item()
        != NACA_BOUND_MESH_SHA256
    ):
        raise ValueError("serialized geometry provenance differs")
    geometry = NACAGeometry.from_mapping(mapping)
    return VerifiedNACAGeometry(
        geometry,
        NACA_BOUND_MESH_SHA256,
        _token=_VERIFIED_CONTEXT_TOKEN,
    )


def _build_naca_geometry_from_mesh(mesh: str | Path | SU2Mesh) -> NACAGeometry:
    """Construct geometry from a mesh; private for focused synthetic tests."""

    parsed = mesh if isinstance(mesh, SU2Mesh) else parse_su2_mesh(mesh)
    if parsed.dimension != 2:
        raise ValueError("NACA PCNO requires a two-dimensional mesh")
    if parsed.element_type_counts != {9: parsed.num_elements}:
        raise ValueError("NACA PCNO requires an all-VTK9 quadrilateral volume mesh")
    if set(parsed.marker_elements) != {"airfoil", "farfield"}:
        raise ValueError("NACA mesh must contain exactly airfoil and farfield markers")
    marker_nodes: dict[str, set[int]] = {}
    marker_edges: dict[str, set[tuple[int, int]]] = {}
    for marker in ("airfoil", "farfield"):
        entries = parsed.marker_elements[marker]
        if not entries:
            raise ValueError(f"NACA marker {marker} is empty")
        nodes: set[int] = set()
        edges: set[tuple[int, int]] = set()
        for vtk_type, element_nodes in entries:
            if vtk_type != 3 or len(element_nodes) != 2:
                raise ValueError(f"NACA marker {marker} must contain only VTK3 lines")
            nodes.update(element_nodes)
            edge = tuple(sorted(element_nodes))
            if edge in edges:
                raise ValueError(f"NACA marker {marker} contains a duplicate edge")
            edges.add(edge)
        marker_nodes[marker] = nodes
        marker_edges[marker] = edges
    if marker_nodes["airfoil"] & marker_nodes["farfield"]:
        raise ValueError("airfoil and farfield marker nodes overlap")

    volume_edges: Counter[tuple[int, int]] = Counter()
    for _, element_nodes in parsed.elements:
        for first, second in zip(
            element_nodes, (*element_nodes[1:], element_nodes[0]), strict=True
        ):
            volume_edges[tuple(sorted((first, second)))] += 1
    if any(count not in (1, 2) for count in volume_edges.values()):
        raise ValueError("quad mesh has a nonmanifold volume edge")
    topological_boundary = {edge for edge, count in volume_edges.items() if count == 1}
    declared_boundary = marker_edges["airfoil"] | marker_edges["farfield"]
    if declared_boundary != topological_boundary:
        raise ValueError(
            "airfoil and farfield markers do not partition the mesh boundary"
        )

    elements = np.asarray(
        [[2, *nodes] for vtk_type, nodes in parsed.elements if vtk_type == 9],
        dtype=np.int64,
    )
    native_nodes = np.asarray(parsed.points, dtype=np.float64)
    # This is the sole allowed definition of m_i under the frozen contract.
    measures = lumped_vertex_areas(parsed).reshape(-1, 1)
    weights = measures / np.sum(measures)
    density = weights / measures

    one_hot = np.zeros((parsed.num_points, 3), dtype=np.float64)
    one_hot[:, 0] = 1.0
    for column, marker in ((1, "airfoil"), (2, "farfield")):
        indices = np.asarray(sorted(marker_nodes[marker]), dtype=np.int64)
        one_hot[indices, 0] = 0.0
        one_hot[indices, column] = 1.0
    if np.count_nonzero(one_hot[:, 0]) == 0:
        raise ValueError("NACA mesh has no interior nodes")

    translated = native_nodes - np.min(native_nodes, axis=0)
    lengths = np.ptp(native_nodes, axis=0)
    directed_edges, gradient_weights, _ = compute_edge_gradient_weights(
        translated,
        elements,
        mesh_type="vertex_centered",
        adjacent_type="element",
        rcond=0.001,
    )
    return NACAGeometry(
        native_coordinates=native_nodes,
        elements=elements,
        node_measures=measures,
        node_weights=weights,
        quadrature_density=density,
        boundary_one_hot=one_hot,
        directed_edges=directed_edges,
        edge_gradient_weights=gradient_weights,
        fourier_lengths=lengths,
        gradient_rcond=0.001,
    )


def build_naca_geometry(context: NACAVerifiedContext) -> VerifiedNACAGeometry:
    """Build geometry only from the mesh bound into verified parent evidence."""

    if not isinstance(context, NACAVerifiedContext):
        raise TypeError("production geometry requires a NACAVerifiedContext")
    _verified_contract_payload(context.contract)
    _, _, mesh_contract, _ = context._payloads(_VERIFIED_CONTEXT_TOKEN)
    mesh_record = context.mesh_record
    mesh_path = context.mesh_path
    digest = _require_digest(mesh_record.get("sha256"), "bound live mesh")
    parsed = _read_with_stable_file_identity(
        mesh_path,
        expected_bytes=mesh_record.get("bytes"),
        expected_sha256=digest,
        label="bound live mesh",
        reader=parse_su2_mesh,
    )
    observed_contract = {
        "dimension": parsed.dimension,
        "num_elements": parsed.num_elements,
        "num_points": parsed.num_points,
        "element_type_counts": {
            str(key): value for key, value in parsed.element_type_counts.items()
        },
        "marker_element_counts": dict(parsed.marker_element_counts),
    }
    if observed_contract != mesh_contract:
        raise ValueError("parsed mesh differs from the verified resource contract")
    return VerifiedNACAGeometry(
        _build_naca_geometry_from_mesh(parsed),
        digest,
        _token=_VERIFIED_CONTEXT_TOKEN,
    )


@dataclass(frozen=True)
class NACANormalization:
    """Train-only state and next-increment scaling for the five evolved fields."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    residual_scale: np.ndarray
    state_rms: np.ndarray

    def __post_init__(self) -> None:
        for name in ("state_mean", "state_scale", "residual_scale", "state_rms"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (NUM_DYNAMIC_FIELDS,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape [5]")
            if name in ("state_scale", "residual_scale") and np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            if name == "state_rms" and np.any(value < 0.0):
                raise ValueError("state_rms must be nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NACANormalization:
        if value.get("schema") != "time_dependent_no.naca_normalization.v1":
            raise ValueError("unsupported NACA normalization schema")
        if value.get("dynamic_fields") != list(NACA_DYNAMIC_FIELDS):
            raise ValueError("normalization dynamic fields differ")
        if (
            value.get("fit_scope") != "train_only_uniform_node"
            or value.get("residual_mean_subtracted") is not False
        ):
            raise ValueError("normalization fit scope differs")
        normalization = cls(
            state_mean=value["state_mean"],
            state_scale=value["state_scale"],
            residual_scale=value["residual_scale"],
            state_rms=value["state_rms"],
        )
        stored_floor = np.asarray(
            value.get("relative_l2_denominator_floor"), dtype=np.float64
        )
        if stored_floor.shape != (NUM_DYNAMIC_FIELDS,) or not np.array_equal(
            stored_floor, normalization.relative_l2_floor
        ):
            raise ValueError("normalization relative-L2 denominator floor differs")
        return normalization

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "time_dependent_no.naca_normalization.v1",
            "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
            "state_mean": self.state_mean.tolist(),
            "state_scale": self.state_scale.tolist(),
            "residual_scale": self.residual_scale.tolist(),
            "state_rms": self.state_rms.tolist(),
            "relative_l2_denominator_floor": self.relative_l2_floor.tolist(),
            "fit_scope": "train_only_uniform_node",
            "residual_mean_subtracted": False,
        }

    @property
    def relative_l2_floor(self) -> np.ndarray:
        return 1.0e-12 * np.maximum(self.state_rms, 1.0)

    def normalize_state(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            mean = torch.tensor(self.state_mean, dtype=value.dtype, device=value.device)
            scale = torch.tensor(
                self.state_scale, dtype=value.dtype, device=value.device
            )
            return (value - mean) / scale
        array = np.asarray(value)
        return (array - self.state_mean) / self.state_scale

    def normalize_residual(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            scale = torch.tensor(
                self.residual_scale, dtype=value.dtype, device=value.device
            )
            return value / scale
        return np.asarray(value) / self.residual_scale

    def decode_residual(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            scale = torch.tensor(
                self.residual_scale, dtype=value.dtype, device=value.device
            )
            return value * scale
        return np.asarray(value) * self.residual_scale


def _validated_frame(value: Any, index: int, expected_nodes: int | None) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.float64:
        raise ValueError(f"training frame {index} must retain native float64 dtype")
    if frame.ndim != 2 or frame.shape[1] != NUM_DYNAMIC_FIELDS:
        raise ValueError(f"training frame {index} must have shape [N,5]")
    if expected_nodes is not None and frame.shape[0] != expected_nodes:
        raise ValueError("training frames do not share the fixed mesh")
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"training frame {index} contains nonfinite values")
    return frame


def fit_naca_normalization(
    frame_loader: Callable[[int], np.ndarray], contract: VerifiedNACAContract
) -> NACANormalization:
    """Fit the frozen uniform-node statistics without reading any nontrain frame."""

    contract_payload = _verified_contract_payload(contract)
    normalization = contract_payload["normalization"]
    frame_first, frame_last = _inclusive_range(
        normalization["fit_frames"], "normalization fit frames"
    )
    residual_first, residual_last = _inclusive_range(
        normalization["fit_transitions"], "normalization fit transitions"
    )
    if residual_first < frame_first or residual_last + 1 > frame_last:
        raise ValueError("normalization residual range escapes train fit frames")

    state_mean = np.zeros(NUM_DYNAMIC_FIELDS, dtype=np.float64)
    state_m2 = np.zeros(NUM_DYNAMIC_FIELDS, dtype=np.float64)
    state_square_sum = np.zeros(NUM_DYNAMIC_FIELDS, dtype=np.float64)
    residual_square_sum = np.zeros(NUM_DYNAMIC_FIELDS, dtype=np.float64)
    state_count = 0
    residual_count = 0
    previous: np.ndarray | None = None
    previous_index: int | None = None
    num_nodes: int | None = None
    for index in range(frame_first, frame_last + 1):
        frame = _validated_frame(frame_loader(index), index, num_nodes)
        if num_nodes is None:
            num_nodes = int(frame.shape[0])
        frame_count = frame.shape[0]
        frame_mean = np.mean(frame, axis=0, dtype=np.float64)
        frame_m2 = np.sum(np.square(frame - frame_mean), axis=0, dtype=np.float64)
        updated_count = state_count + frame_count
        mean_delta = frame_mean - state_mean
        state_m2 += frame_m2 + np.square(mean_delta) * (
            state_count * frame_count / updated_count
        )
        state_mean += mean_delta * (frame_count / updated_count)
        state_square_sum += np.sum(np.square(frame), axis=0, dtype=np.float64)
        state_count = updated_count
        if (
            previous is not None
            and previous_index is not None
            and residual_first <= previous_index <= residual_last
        ):
            residual = np.subtract(frame, previous, dtype=np.float64)
            residual_square_sum += np.sum(np.square(residual), axis=0, dtype=np.float64)
            residual_count += frame.shape[0]
        previous = frame
        previous_index = index
    if state_count < 1 or residual_count < 1:
        raise ValueError("normalization fit received no train samples")
    state_second = state_square_sum / state_count
    state_variance = np.maximum(state_m2 / state_count, 0.0)
    state_rms = np.sqrt(state_second)
    state_floor = 1.0e-8 * np.maximum(state_rms, 1.0)
    state_scale = np.maximum(np.sqrt(state_variance), state_floor)
    residual_rms = np.sqrt(residual_square_sum / residual_count)
    residual_scale = np.maximum(residual_rms, 1.0e-6 * state_scale)
    return NACANormalization(state_mean, state_scale, residual_scale, state_rms)


def validate_role_access(
    contract: VerifiedNACAContract,
    role: str,
    center_indices: Iterable[int],
    *,
    horizon_steps: int = 1,
) -> tuple[int, ...]:
    """Validate open-role leakage; protected roles have no bypass in this module."""

    contract_payload = _verified_contract_payload(contract)
    record = _open_role_record(contract_payload, role)
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps < 1
    ):
        raise ValueError("horizon_steps must be a positive integer")
    first, last = _inclusive_range(
        record["owned_frame_indices_inclusive"], f"{role} owned frames"
    )
    centers = _strict_indices(center_indices, "center indices")
    for center in centers:
        if center - 1 < first or center + horizon_steps > last:
            raise ValueError(
                f"{role} center {center} with horizon {horizon_steps} leaks"
            )
    return centers


class NACABDF2Dataset(Dataset):
    """Float64-backed complete-BDF2 residual presentations on one fixed mesh."""

    def __init__(
        self,
        states: np.ndarray,
        frame_indices: Sequence[int],
        transition_center_indices: Sequence[int],
        normalization: NACANormalization,
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_CONTEXT_TOKEN:
            raise TypeError("use build_naca_bdf2_dataset for production data")
        array = np.asarray(states)
        indices = _strict_indices(frame_indices, "frame indices")
        if array.dtype != np.float64 or array.ndim != 3 or array.shape[-1] != 5:
            raise ValueError("stored states must retain float64 shape [T,N,5]")
        if array.shape[0] != len(indices):
            raise ValueError("frame indices must uniquely align with stored states")
        if any(right != left + 1 for left, right in pairwise(indices)):
            raise ValueError("stored role frames must be contiguous and nonduplicated")
        centers = _strict_indices(
            transition_center_indices, "transition center indices"
        )
        if not isinstance(normalization, NACANormalization):
            raise TypeError("a NACANormalization is required")
        positions = {index: offset for offset, index in enumerate(indices)}
        for center in centers:
            if any(
                index not in positions for index in (center - 1, center, center + 1)
            ):
                raise ValueError(f"BDF2 transition {center} lacks a required frame")
        self.states = array
        self.frame_indices = indices
        self.center_indices = centers
        self.normalization = normalization
        self._positions = positions

    def __len__(self) -> int:
        return len(self.center_indices)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        center = self.center_indices[item]
        previous64 = _validated_frame(
            self.states[self._positions[center - 1]], center - 1, self.states.shape[1]
        )
        current64 = _validated_frame(
            self.states[self._positions[center]], center, self.states.shape[1]
        )
        target64 = _validated_frame(
            self.states[self._positions[center + 1]], center + 1, self.states.shape[1]
        )
        residual64 = np.subtract(target64, current64, dtype=np.float64)
        normalized64 = self.normalization.normalize_residual(residual64)
        return (
            torch.from_numpy(np.array(previous64, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(current64, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(normalized64, dtype=np.float32, copy=True)),
            center,
        )


def _build_synthetic_bdf2_dataset(
    states: np.ndarray,
    frame_indices: Sequence[int],
    transition_center_indices: Sequence[int],
    normalization: NACANormalization,
) -> NACABDF2Dataset:
    return NACABDF2Dataset(
        states,
        frame_indices,
        transition_center_indices,
        normalization,
        _token=_VERIFIED_CONTEXT_TOKEN,
    )


def build_naca_bdf2_dataset(
    contract: VerifiedNACAContract,
    geometry: VerifiedNACAGeometry,
    role: str,
    states: np.ndarray,
    frame_indices: Sequence[int],
    normalization: NACANormalization,
) -> NACABDF2Dataset:
    """Construct the exact 240-frame/238-center dataset for one open role."""

    contract_payload = _verified_contract_payload(contract)
    if not isinstance(geometry, VerifiedNACAGeometry):
        raise TypeError("production datasets require VerifiedNACAGeometry")
    native_geometry = geometry._geometry(_VERIFIED_CONTEXT_TOKEN)
    record = _open_role_record(contract_payload, role)
    first, last = _inclusive_range(
        record["owned_frame_indices_inclusive"], f"{role} owned frames"
    )
    expected_frames = tuple(range(first, last + 1))
    observed_frames = _strict_indices(frame_indices, "frame indices")
    if observed_frames != expected_frames or len(expected_frames) != 240:
        raise ValueError(f"{role} dataset must contain its exact 240 ordered frames")
    center_first, center_last = _inclusive_range(
        record["dense_transition_center_indices_inclusive"],
        f"{role} dense transitions",
    )
    centers = tuple(range(center_first, center_last + 1))
    if len(centers) != 238 or record.get("dense_transition_count") != 238:
        raise ValueError(f"{role} dataset must contain exactly 238 ordered centers")
    array = np.asarray(states)
    if array.ndim != 3 or array.shape[1] != native_geometry.num_nodes:
        raise ValueError("role states do not match the verified fixed mesh")
    return NACABDF2Dataset(
        array,
        observed_frames,
        centers,
        normalization,
        _token=_VERIFIED_CONTEXT_TOKEN,
    )


class NACAPCNOResidual(nn.Module):
    """Thin complete-BDF2 five-residual wrapper around the generic PCNO."""

    def __init__(
        self,
        *,
        fourier_lengths: Sequence[float],
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        lengths = tuple(float(value) for value in fourier_lengths)
        if len(lengths) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("fourier_lengths must contain two positive finite values")
        modes = compute_Fourier_modes(2, [8, 8], list(lengths))
        self.backbone = PCNO(
            2,
            torch.as_tensor(modes, dtype=torch.float32),
            nmeasures=1,
            layers=[128, 128, 128, 128, 128],
            fc_dim=128,
            in_dim=NUM_INPUT_FEATURES,
            out_dim=NUM_DYNAMIC_FIELDS,
            act="gelu",
        )
        self.fourier_lengths = lengths
        if zero_initialize:
            self.zero_initialize_output_head()

    def zero_initialize_output_head(self) -> None:
        """Make the deployed recurrent map exact persistence at initialization."""

        nn.init.zeros_(self.backbone.fc2.weight)
        nn.init.zeros_(self.backbone.fc2.bias)

    def model_config(self) -> dict[str, Any]:
        return {
            "model": "NACAPCNOResidual",
            "backbone": "pcno.PCNO",
            "history": "complete BDF2 pair",
            "input_channels": NUM_INPUT_FEATURES,
            "output_channels": NUM_DYNAMIC_FIELDS,
            "input_feature_names": list(NACA_INPUT_FEATURE_NAMES),
            "fourier_modes": [8, 8],
            "fourier_lengths": list(self.fourier_lengths),
            "nmeasures": 1,
            "layers": [128, 128, 128, 128, 128],
            "projection_width": 128,
            "activation": "gelu",
            "gradient_branch": True,
            "output": "normalized five-field next-state residual",
            "zero_initialization": (
                "set both backbone.fc2.weight and backbone.fc2.bias exactly to zero "
                "before the first optimizer step"
            ),
        }

    def prepare_fourier_tensors(
        self, geometry_batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        return self.backbone.prepare_fourier_tensors(
            geometry_batch["nodes"], geometry_batch["node_weights"]
        )

    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: Mapping[str, torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != NUM_INPUT_FEATURES:
            raise ValueError("NACA PCNO features must have shape [B,N,16]")
        expected = features.shape[:2]
        if geometry_batch["nodes"].shape[:2] != expected:
            raise ValueError("fixed geometry does not align with model features")
        auxiliary = (
            geometry_batch["node_mask"],
            geometry_batch["nodes"],
            geometry_batch["node_weights"],
            geometry_batch["directed_edges"],
            geometry_batch["edge_gradient_weights"],
        )
        return self.backbone(
            features,
            auxiliary,
            fourier_tensors=fourier_tensors,
        )


def build_naca_pcno(
    contract: VerifiedNACAContract,
    geometry: VerifiedNACAGeometry,
    *,
    zero_initialize: bool = True,
) -> NACAPCNOResidual:
    """Build exactly the architecture frozen by the baseline contract."""

    _verified_contract_payload(contract)
    if not isinstance(geometry, VerifiedNACAGeometry):
        raise TypeError("production PCNO requires VerifiedNACAGeometry")
    native_geometry = geometry._geometry(_VERIFIED_CONTEXT_TOKEN)
    if zero_initialize is not True:
        raise ValueError(
            "the frozen baseline requires exact zero output initialization"
        )
    model = NACAPCNOResidual(
        fourier_lengths=native_geometry.fourier_lengths,
        zero_initialize=zero_initialize,
    )
    validate_naca_model_config(model.model_config(), contract, geometry)
    return model


def _expected_model_config(geometry: NACAGeometry) -> dict[str, Any]:
    return {
        "model": "NACAPCNOResidual",
        "backbone": "pcno.PCNO",
        "history": "complete BDF2 pair",
        "input_channels": NUM_INPUT_FEATURES,
        "output_channels": NUM_DYNAMIC_FIELDS,
        "input_feature_names": list(NACA_INPUT_FEATURE_NAMES),
        "fourier_modes": [8, 8],
        "fourier_lengths": geometry.fourier_lengths.tolist(),
        "nmeasures": 1,
        "layers": [128, 128, 128, 128, 128],
        "projection_width": 128,
        "activation": "gelu",
        "gradient_branch": True,
        "output": "normalized five-field next-state residual",
        "zero_initialization": (
            "set both backbone.fc2.weight and backbone.fc2.bias exactly to zero "
            "before the first optimizer step"
        ),
    }


def validate_naca_model_config(
    value: Mapping[str, Any],
    contract: VerifiedNACAContract,
    geometry: VerifiedNACAGeometry,
) -> None:
    """Reject checkpoints whose model configuration is not the frozen map."""

    _verified_contract_payload(contract)
    if not isinstance(geometry, VerifiedNACAGeometry):
        raise TypeError("checkpoint validation requires VerifiedNACAGeometry")
    expected = _expected_model_config(geometry._geometry(_VERIFIED_CONTEXT_TOKEN))
    if dict(value) != expected:
        raise ValueError("checkpoint model configuration differs from the frozen PCNO")


def build_input(
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
) -> torch.Tensor:
    """Assemble the exact 6 static + 5 previous + 5 current feature order."""

    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[-1] != 5:
        raise ValueError("previous and current states must share shape [B,N,5]")
    if previous.dtype != torch.float32 or current.dtype != torch.float32:
        raise ValueError("model-facing BDF2 states must be float32")
    static = geometry_batch.get("static_features")
    if static is None or static.shape != previous.shape[:2] + (NUM_STATIC_FEATURES,):
        raise ValueError("static geometry features must have shape [B,N,6]")
    if static.device != previous.device or current.device != previous.device:
        raise ValueError("states and geometry must share a device")
    features = torch.cat(
        (
            static.to(dtype=previous.dtype),
            normalization.normalize_state(previous),
            normalization.normalize_state(current),
        ),
        dim=-1,
    )
    if features.shape[-1] != NUM_INPUT_FEATURES:
        raise AssertionError("NACA model input layout is inconsistent")
    return features


def predict_normalized_residual(
    model: NACAPCNOResidual,
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Evaluate one normalized residual with optional precomputed Fourier bases."""

    features = build_input(previous, current, geometry_batch, normalization)
    return model(
        features,
        geometry_batch,
        fourier_tensors=fourier_tensors,
    )


def recurrent_step(
    model: NACAPCNOResidual,
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``(u[n-1],u[n]) -> (u[n],u[n] + decoded residual)``."""

    normalized_residual = predict_normalized_residual(
        model,
        previous,
        current,
        geometry_batch,
        normalization,
        fourier_tensors=fourier_tensors,
    )
    if normalized_residual.shape != current.shape:
        raise ValueError("model residual shape differs from the current state")
    next_state = current + normalization.decode_residual(normalized_residual)
    return current, next_state

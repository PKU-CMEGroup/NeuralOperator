#!/usr/bin/env python3
"""Run the registered L4A-004 conservative local-Haar capacity preflight.

The command fits one deterministic training-only discontinuous chart and
evaluates reconstruction on the frozen position-OOD validation states. It has
no latent transition, rollout, test access, per-state optimization, or filter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_euler2d_latent_code_reachability import (  # noqa: E402
    _aggregate_rows,
    _reconstruction_row,
)
from scripts.time_dependent_no.train_euler2d_latent_representation import (  # noqa: E402
    SMOKE_FRAMES,
    _append_jsonl,
    _load_geometry,
    _load_state_batch,
    _prepare_output,
    _smoke_data_preflight,
    _write_json,
    _write_rows,
    audit_inputs,
)
from utility.time_dependent_no.latent_representation_2d import (  # noqa: E402
    LINE4_LOCAL_HAAR_SCHEMA,
    fit_conservative_local_haar_atlas,
    sha256_file,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DShardStore,
)

SCHEMA = "line4a_conservative_local_haar_capacity_v1"
CONTROL_SCHEMA = "line4a_frozen_decoder_code_reachability_v1"
CONTROL_SUMMARY_SHA256 = (
    "e9577b437824069ff0db9dc28b9072368754eac2ba14453ef4039a41083a798a"
)
DETAIL_CHANNELS = 16
MAXIMUM_HAAR_LEVEL = 3
RECONSTRUCTION_L2_LIMIT = 0.0021
FRONT_RELATIVE_TOLERANCE = 0.05
MAXIMUM_CONTROL_OVERSHOOT = 0.0708514
MOMENT_RELATIVE_L2_LIMIT = 1.0e-5
CPU_HOUR_LIMIT = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("preflight", "capacity"),
        default="preflight",
    )
    return parser.parse_args(argv)


def _load_control_artifact(control_dir: Path) -> dict[str, Any]:
    summary_path = control_dir / "summary.json"
    ledger_path = control_dir / "artifact_ledger.json"
    physical_path = control_dir / "physical_states.npz"
    if sha256_file(summary_path) != CONTROL_SUMMARY_SHA256:
        raise ValueError("L4A-003 control summary digest mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != CONTROL_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("classification") != "decoder_manifold_rejected"
    ):
        raise ValueError("L4A-003 control result is not the frozen rejection")
    if ledger.get("schema") != CONTROL_SCHEMA:
        raise ValueError("L4A-003 ledger schema mismatch")
    files = ledger.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("L4A-003 ledger lacks file digests")
    for name, record in files.items():
        path = control_dir / str(name)
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"L4A-003 artifact digest mismatch for {name}")
    with np.load(physical_path, allow_pickle=False) as archive:
        required = {"schema", "split", "trajectories", "frames", "target"}
        if not required.issubset(archive.files):
            raise ValueError("L4A-003 physical-state archive is incomplete")
        if str(archive["schema"].item()) != CONTROL_SCHEMA:
            raise ValueError("L4A-003 physical-state schema mismatch")
        targets = np.array(archive["target"], dtype=np.float32, copy=True)
        trajectories = [str(value) for value in archive["trajectories"].tolist()]
        frames = [int(value) for value in archive["frames"].tolist()]
    if targets.shape != (10, 25000, 4):
        raise ValueError("L4A-003 target bank has the wrong shape")
    return {
        "summary": summary,
        "ledger": ledger,
        "summary_sha256": CONTROL_SUMMARY_SHA256,
        "ledger_sha256": sha256_file(ledger_path),
        "physical_states_sha256": sha256_file(physical_path),
        "targets": targets,
        "trajectories": trajectories,
        "frames": frames,
    }


def _preflight(
    store: PCNOEuler2DShardStore,
    audit: Mapping[str, Any],
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    smoke = _smoke_data_preflight(
        store,
        train_keys,
        validation_keys,
        device=torch.device("cpu"),
    )
    selected_train = list(smoke["train_trajectories"])
    selected_validation = list(smoke["validation_trajectories"])
    summary = control["summary"]
    configuration = summary.get("configuration", {})
    if configuration.get("training_trajectories_for_code_scale") != selected_train:
        raise ValueError("L4A-003 and L4A-004 training cohorts differ")
    if configuration.get("validation_trajectories") != selected_validation:
        raise ValueError("L4A-003 and L4A-004 validation cohorts differ")
    if configuration.get("frames") != list(SMOKE_FRAMES):
        raise ValueError("L4A-003 and L4A-004 frame contracts differ")
    if summary.get("audit", {}).get("data_manifest_digest") != audit.get(
        "data_manifest_digest"
    ):
        raise ValueError("L4A-003 and L4A-004 data manifests differ")
    expected_trajectories = [key for key in selected_validation for _ in SMOKE_FRAMES]
    expected_frames = [
        int(frame) for _ in selected_validation for frame in SMOKE_FRAMES
    ]
    if control["trajectories"] != expected_trajectories:
        raise ValueError("L4A-003 target trajectory ordering changed")
    if control["frames"] != expected_frames:
        raise ValueError("L4A-003 target frame ordering changed")
    current_targets = _load_state_batch(
        store,
        list(zip(expected_trajectories, expected_frames, strict=True)),
        torch.device("cpu"),
    ).numpy()
    if not np.array_equal(current_targets, control["targets"]):
        raise ValueError("L4A-003 target bytes differ from staged validation states")
    return {
        "schema": SCHEMA,
        "stage": "preflight",
        "status": "passed",
        "audit": dict(audit),
        "smoke_data_preflight": smoke,
        "selected_training_trajectories": selected_train,
        "selected_validation_trajectories": selected_validation,
        "frames": list(SMOKE_FRAMES),
        "training_states_for_selection": len(selected_train) * len(SMOKE_FRAMES),
        "validation_states": len(expected_trajectories),
        "control_summary_sha256": control["summary_sha256"],
        "control_ledger_sha256": control["ledger_sha256"],
        "control_physical_states_sha256": control["physical_states_sha256"],
        "targets_bitwise_equal": True,
        "test_arrays_read": False,
        "transition_present": False,
        "rollout_present": False,
        "analysis_applied": False,
    }


def classify_local_haar_capacity(
    aggregate: Mapping[str, Any],
    control_aggregate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    overshoot_limit = min(
        MAXIMUM_CONTROL_OVERSHOOT,
        float(control_aggregate["maximum_density_pressure_overshoot"]),
    )
    gates = {
        "mean_relative_l2_at_most_0p0021": (
            float(aggregate["mean_relative_l2"]) <= RECONSTRUCTION_L2_LIMIT
        ),
        "all_states_raw_admissible": (
            float(aggregate["minimum_admissible_fraction"]) >= 1.0
        ),
        "mean_shock_strength_within_5_percent": (
            abs(float(aggregate["mean_shock_strength_ratio"]) - 1.0)
            <= FRONT_RELATIVE_TOLERANCE
        ),
        "mean_shock_thickness_within_5_percent": (
            abs(float(aggregate["mean_shock_thickness_ratio"]) - 1.0)
            <= FRONT_RELATIVE_TOLERANCE
        ),
        "maximum_overshoot_not_increased": (
            float(aggregate["maximum_density_pressure_overshoot"])
            <= overshoot_limit + 1.0e-8
        ),
        "maximum_token_moment_relative_l2_at_most_1e_5": (
            float(aggregate["maximum_token_moment_relative_l2"])
            <= MOMENT_RELATIVE_L2_LIMIT
        ),
        "maximum_global_budget_relative_l2_at_most_1e_5": (
            float(aggregate["maximum_global_budget_relative_l2_mean"])
            <= MOMENT_RELATIVE_L2_LIMIT
        ),
        "exact_finite_5000_scalar_rank_contract": (
            int(contract.get("latent_size", -1)) == 5000
            and int(contract.get("training_selected_detail_channels", -1))
            == DETAIL_CHANNELS
            and int(contract.get("minimum_local_basis_rank", -1)) * 4 >= DETAIL_CHANNELS
            and math.isfinite(
                float(contract.get("minimum_selected_training_energy", math.nan))
            )
        ),
    }
    return (
        "capacity_only_pass" if all(gates.values()) else "local_haar_capacity_rejected",
        gates,
    )


def _write_capacity_artifacts(
    output_dir: Path,
    *,
    trajectories: Sequence[str],
    frames: Sequence[int],
    targets: np.ndarray,
    decoded: np.ndarray,
    codes: np.ndarray,
    selected_modes: np.ndarray,
    selected_directions: np.ndarray,
    selected_training_energy: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / "physical_states.npz",
        schema=np.asarray(SCHEMA),
        split=np.asarray("validation"),
        trajectories=np.asarray(trajectories),
        frames=np.asarray(frames, dtype=np.int64),
        target=targets,
        local_haar_decoded=decoded,
    )
    np.savez_compressed(
        output_dir / "latent_states.npz",
        schema=np.asarray(SCHEMA),
        split=np.asarray("validation"),
        trajectories=np.asarray(trajectories),
        frames=np.asarray(frames, dtype=np.int64),
        code=codes,
        selected_modes=selected_modes,
        selected_directions=selected_directions,
        selected_training_energy=selected_training_energy,
    )


def _run_capacity(
    args: argparse.Namespace,
    store: PCNOEuler2DShardStore,
    audit: Mapping[str, Any],
    normalization: Euler2DNormalization,
    control: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    selected_train = list(preflight["selected_training_trajectories"])
    selected_validation = list(preflight["selected_validation_trajectories"])
    geometry = _load_geometry(store, selected_train[0], torch.device("cpu"))
    edges = np.array(
        store.array(selected_validation[0], "edges"),
        dtype=np.int64,
        copy=True,
    )
    training_items = [
        (key, int(frame)) for key in selected_train for frame in SMOKE_FRAMES
    ]
    validation_items = [
        (key, int(frame)) for key in selected_validation for frame in SMOKE_FRAMES
    ]

    start_time = perf_counter()
    training_states = _load_state_batch(
        store,
        training_items,
        torch.device("cpu"),
    )
    atlas, fit_diagnostics = fit_conservative_local_haar_atlas(
        training_states,
        geometry,
        normalization,
        detail_channels=DETAIL_CHANNELS,
        maximum_level=MAXIMUM_HAAR_LEVEL,
    )
    contract = atlas.contract()
    if contract["latent_size"] != 5000:
        raise RuntimeError("L4A-004 latent budget changed")
    validation_states = _load_state_batch(
        store,
        validation_items,
        torch.device("cpu"),
    )
    with torch.no_grad():
        codes = atlas.encode(validation_states, geometry)
        decoded = atlas.decode(codes, geometry)

    rows: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    for index, (trajectory, frame) in enumerate(validation_items):
        row = _reconstruction_row(
            variant="local_haar",
            trajectory=trajectory,
            frame=frame,
            prediction=decoded[index : index + 1],
            target=validation_states[index : index + 1],
            geometry=geometry,
            normalization=normalization,
            edges=edges,
            displacement_rms=0.0,
            per_state_fitting=False,
        )
        row["training_selected_dictionary"] = True
        row["encoder_type"] = "single_weighted_projection"
        rows.append(row)
        _append_jsonl(
            metrics_path,
            {
                **row,
                "test_arrays_read": False,
                "transition_present": False,
                "analysis_applied": False,
            },
        )

    wall_seconds = perf_counter() - start_time
    aggregate = _aggregate_rows(rows)["local_haar"]
    control_reconstruction = control["summary"]["reconstruction"]
    control_aggregate = control_reconstruction["amortized_encoder"]
    classification, gates = classify_local_haar_capacity(
        aggregate,
        control_aggregate,
        contract,
    )
    _write_rows(args.output_dir / "reconstruction_rows.csv", rows)
    _write_rows(
        args.output_dir / "summary.csv",
        [
            {"variant": "amortized_encoder_control", **control_aggregate},
            {
                "variant": "fitted_code_control",
                **control_reconstruction["fitted_code"],
            },
            {"variant": "local_haar", **aggregate},
        ],
    )
    target_numpy = validation_states.numpy()
    decoded_numpy = decoded.numpy()
    code_numpy = codes.numpy()
    trajectories = [key for key, _ in validation_items]
    frames = [frame for _, frame in validation_items]
    _write_capacity_artifacts(
        args.output_dir,
        trajectories=trajectories,
        frames=frames,
        targets=target_numpy,
        decoded=decoded_numpy,
        codes=code_numpy,
        selected_modes=atlas.selected_modes.cpu().numpy(),
        selected_directions=atlas.selected_directions.cpu().numpy(),
        selected_training_energy=atlas.selected_training_energy.cpu().numpy(),
    )

    cpu_hours = wall_seconds / 3600.0
    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "classification": classification,
        "hypothesis": (
            "at the same 5000-scalar token budget and exact token moments, a "
            "training-selected discontinuous local Haar chart satisfies the "
            "sharp physical reconstruction gates that reject the smooth decoder"
        ),
        "smallest_matched_control": (
            "the frozen L4A-003 smooth conservative decoder on the exact same "
            "10 validation states; its per-state fitted code is the stronger "
            "capacity control and its amortized code sets the overshoot ceiling"
        ),
        "audit": dict(audit),
        "control": {
            "schema": CONTROL_SCHEMA,
            "summary_sha256": control["summary_sha256"],
            "ledger_sha256": control["ledger_sha256"],
            "physical_states_sha256": control["physical_states_sha256"],
            "targets_bitwise_equal": True,
        },
        "configuration": {
            "training_trajectories": selected_train,
            "validation_trajectories": selected_validation,
            "frames": list(SMOKE_FRAMES),
            "training_states": len(training_items),
            "validation_states": len(validation_items),
            "device": "cpu",
            "maximum_haar_level": MAXIMUM_HAAR_LEVEL,
            "detail_channels": DETAIL_CHANNELS,
        },
        "representation_contract": contract,
        "fit_diagnostics": fit_diagnostics,
        "reconstruction": {
            "amortized_encoder_control": control_aggregate,
            "fitted_code_control": control_reconstruction["fitted_code"],
            "local_haar": aggregate,
        },
        "promotion_gates": gates,
        "capacity_pass": classification == "capacity_only_pass",
        "promotion_eligible": False,
        "closure_evidence": False,
        "autonomous_forecast_evidence": False,
        "assimilation_evidence": False,
        "stage_contract": {
            "training_only_dictionary_selection": True,
            "validation_encoding": "single_weighted_projection",
            "per_state_fitting": False,
            "front_variables": 0,
            "front_fit": False,
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
            "transition_present": False,
            "rollout_present": False,
            "test_arrays_read": False,
            "analysis_applied": False,
            "forecast_evidence": False,
        },
        "cost": {
            "wall_seconds": wall_seconds,
            "cpu_hours": cpu_hours,
            "cpu_hour_limit": CPU_HOUR_LIMIT,
            "within_budget": cpu_hours < CPU_HOUR_LIMIT,
            "gpu_hours": 0.0,
        },
        "failure_interpretation": (
            "all physical capacity gates pass; only a separately authorized "
            "zero-training closure and conditioning preflight may be proposed"
            if classification == "capacity_only_pass"
            else "the fixed local discontinuous chart fails at least one joint "
            "physical gate; stop it without changing levels, atoms, lattice, "
            "selection objective, or cohort"
        ),
        "next_decision": (
            "review this capacity artifact; serious representation training, "
            "transition training, test access, and filtering remain blocked"
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)

    artifact_names = (
        "preflight.json",
        "metrics.jsonl",
        "reconstruction_rows.csv",
        "summary.csv",
        "summary.json",
        "physical_states.npz",
        "latent_states.npz",
    )
    ledger = {
        "schema": SCHEMA,
        "representation_schema": LINE4_LOCAL_HAAR_SCHEMA,
        "encoder_sha256": atlas.atlas_sha256,
        "decoder_sha256": atlas.atlas_sha256,
        "geometry_sha256": contract["geometry_sha256"],
        "normalization_digest": audit["normalization_digest"],
        "split": "validation",
        "training_trajectories": selected_train,
        "validation_trajectories": selected_validation,
        "frames": list(SMOKE_FRAMES),
        "physical_states_file": "physical_states.npz",
        "latent_states_file": "latent_states.npz",
        "valid_length": len(validation_items),
        "failure_cause": classification,
        "observation_contract": None,
        "ensemble_seeds": [],
        "analysis_applied": False,
        "diagnostic_interventions": {
            "training_selected_dictionary": True,
            "per_state_fitting": False,
        },
        "deployment_interventions": {
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
        },
        "transition_present": False,
        "rollout_present": False,
        "test_arrays_read": False,
        "forecast_evidence": False,
        "files": {
            name: {"sha256": sha256_file(args.output_dir / name)}
            for name in artifact_names
        },
    }
    _write_json(args.output_dir / "artifact_ledger.json", ledger)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _prepare_output(args.output_dir)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=4)
    try:
        audit, normalization, train_keys, validation_keys = audit_inputs(
            store,
            handoff_path=args.handoff,
            normalization_path=args.normalization,
        )
        control = _load_control_artifact(args.control_dir)
        preflight = _preflight(
            store,
            audit,
            train_keys,
            validation_keys,
            control,
        )
        _write_json(args.output_dir / "preflight.json", preflight)
        if args.stage == "preflight":
            return
        _run_capacity(
            args,
            store,
            audit,
            normalization,
            control,
            preflight,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()

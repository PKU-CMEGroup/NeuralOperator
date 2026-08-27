"""Run the registered M1-Q1-R2 candidate-reference qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from scripts.time_dependent_no.run_m1_kolmogorov_q1_qualification import (
    INITIAL_SEEDS,
    PERTURBATION_SEEDS,
    Q1Settings,
    _band_perturbation,
    _initial_state,
    _process_repeatability,
    _sha256,
    _state_metadata,
    _time_refinement,
)
from scripts.time_dependent_no.run_m1_kolmogorov_q1_r1_diagnostic import (
    R1Settings,
    _run_spatial,
    _verify_parent_packet,
    _verify_spatial_parent_replay,
)
from scripts.time_dependent_no.run_m1_kolmogorov_q1_r1_diagnostic import (
    _full_contract as _r1_full_contract,
)
from scripts.time_dependent_no.run_m1_kolmogorov_q1_r1_diagnostic import (
    _quick_contract as _r1_quick_contract,
)
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
    resize_dealiased_vorticity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "M1-KF-Q1-R2-REF-20260827B-SPATIAL"
R1_SOURCE_COMMIT = "26e8ef870b7b93f46326d5be451ebb4d2023f234"
R1_MIX_RUN_ID = "M1-KF-Q1-R1-MIX-20260826A"
R1_MIX_RESULT_SHA256 = (
    "fe3e0232d5ea65ba8b96148b6828ac2f5fa599af90978ef6b83b076ba97e6335"
)
R1_MIX_MANIFEST_SHA256 = (
    "c3c91f915716c08c0c9045b097cb45578e3974c7bee8e475556718807f651a44"
)
R1_SPATIAL_RUN_ID = "M1-KF-Q1-R1-SPAT-20260826A"
R1_SPATIAL_RESULT_SHA256 = (
    "31dd4bdfa2e22c2bd3cdaea69201361a1e42d7375e24a677751877cc9834938d"
)
R1_SPATIAL_MANIFEST_SHA256 = (
    "22de2039374be156fde278b1b1fb5eeaf4c02b0f36de1582537bda19b63a63a5"
)
R1_MIX_ROOT = (
    REPO_ROOT / "artifacts/time_dependent_no/m1_kolmogorov_q1_r1_mix_20260826a"
)
R1_SPATIAL_ROOT = (
    REPO_ROOT / "artifacts/time_dependent_no/m1_kolmogorov_q1_r1_spat_20260826a"
)
SOURCE_PATHS = (
    "utility/time_dependent_no/kolmogorov_reference.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_qualification.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_r1_diagnostic.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_r2_diagnostic.py",
    "tests/time_dependent_no/test_kolmogorov_reference.py",
    "tests/time_dependent_no/test_kolmogorov_q1_r1_diagnostic.py",
    "tests/time_dependent_no/test_kolmogorov_q1_r2_diagnostic.py",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_TRACKER.md",
)
SPATIAL_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-13
TEMPORAL_WORKERS = 3


def _full_contract() -> tuple[
    KolmogorovReferenceConfig, R1Settings, Q1Settings
]:
    base_config, r1_settings = _r1_full_contract()
    spatial_settings = replace(
        r1_settings,
        spatial_resolutions=(128, 256, 512),
        workers=3,
    ).validated()
    temporal_settings = Q1Settings(
        burnin_calls=512,
        extended_burnin_calls=512,
        observation_calls=0,
        path_horizon=16,
        structure_horizon=64,
        spatial_resolution=512,
        initial_rms=4.0,
        time_steps=(0.002, 0.001, 0.0005),
        perturbation_bands=spatial_settings.perturbation_bands,
    )
    return base_config, spatial_settings, temporal_settings


def _quick_contract() -> tuple[
    KolmogorovReferenceConfig, R1Settings, Q1Settings
]:
    base_config, r1_settings = _r1_quick_contract()
    spatial_settings = replace(
        r1_settings,
        spatial_resolutions=(36, 72, 144),
        workers=2,
    ).validated()
    temporal_settings = Q1Settings(
        burnin_calls=spatial_settings.reproduction_burnin,
        extended_burnin_calls=spatial_settings.reproduction_burnin,
        observation_calls=0,
        path_horizon=2,
        structure_horizon=4,
        spatial_resolution=144,
        initial_rms=spatial_settings.initial_rms,
        time_steps=(0.002, 0.001, 0.0005),
        perturbation_bands=spatial_settings.perturbation_bands,
    )
    return base_config, spatial_settings, temporal_settings


def _verify_packet(
    root: Path,
    *,
    expected_run_id: str,
    expected_result_sha256: str,
    expected_manifest_sha256: str,
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "result.json"
    manifest_path = root / "artifact_manifest.json"
    if _sha256(result_path) != expected_result_sha256:
        raise RuntimeError(f"result hash mismatch: {result_path}")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError(f"manifest hash mismatch: {manifest_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != expected_run_id:
        raise RuntimeError(f"result identity mismatch: {root}")
    if manifest.get("run_id") != expected_run_id:
        raise RuntimeError(f"manifest identity mismatch: {root}")
    if manifest.get("source_commit") != expected_source_commit:
        raise RuntimeError(f"source commit mismatch: {root}")
    for relative, expected in manifest.get("artifacts", {}).items():
        member = (root / relative).resolve()
        if root.resolve() not in member.parents:
            raise RuntimeError(f"artifact path escapes packet: {relative}")
        if _sha256(member) != expected:
            raise RuntimeError(f"artifact hash mismatch: {member}")
    for relative, expected in manifest.get("sources", {}).items():
        payload = subprocess.run(
            ["git", "show", f"{expected_source_commit}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f"commit-source hash mismatch: {relative}")
    return result, {
        "run_id": expected_run_id,
        "source_commit": expected_source_commit,
        "result_sha256": expected_result_sha256,
        "artifact_manifest_sha256": expected_manifest_sha256,
        "artifact_count": len(manifest.get("artifacts", {})),
        "source_count": len(manifest.get("sources", {})),
        "all_artifacts_match": True,
        "all_commit_sources_match": True,
    }


def _verify_r1_packets() -> tuple[dict[str, Any], dict[str, Any]]:
    mixing_result, mixing_binding = _verify_packet(
        R1_MIX_ROOT,
        expected_run_id=R1_MIX_RUN_ID,
        expected_result_sha256=R1_MIX_RESULT_SHA256,
        expected_manifest_sha256=R1_MIX_MANIFEST_SHA256,
        expected_source_commit=R1_SOURCE_COMMIT,
    )
    spatial_result, spatial_binding = _verify_packet(
        R1_SPATIAL_ROOT,
        expected_run_id=R1_SPATIAL_RUN_ID,
        expected_result_sha256=R1_SPATIAL_RESULT_SHA256,
        expected_manifest_sha256=R1_SPATIAL_MANIFEST_SHA256,
        expected_source_commit=R1_SOURCE_COMMIT,
    )
    return (
        {"mixing": mixing_result, "spatial": spatial_result},
        {"mixing": mixing_binding, "spatial": spatial_binding},
    )


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered source is missing: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _verify_source_binding(source_commit: str, *, quick: bool) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", *SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if quick:
        return {
            "mode": "quick_allows_uncommitted_sources",
            "head": head,
            "requested_source_commit": source_commit,
            "source_status": status.splitlines(),
        }
    if source_commit != head:
        raise ValueError(
            f"--source-commit must equal current HEAD {head}, got {source_commit}"
        )
    if status:
        raise RuntimeError("registered source paths differ from HEAD:\n" + status)
    return {
        "mode": "exact_clean_head",
        "head": head,
        "requested_source_commit": source_commit,
        "source_status": [],
    }


def _input_seed_worker(
    config_values: dict[str, object],
    settings: R1Settings,
    index: int,
) -> dict[str, object]:
    stepper = KolmogorovReferenceStepper(
        KolmogorovReferenceConfig(**config_values).validated()
    )
    seed = INITIAL_SEEDS[index]
    initial = _initial_state(stepper, seed, settings.initial_rms)
    clean = initial
    for _ in range(settings.reproduction_burnin):
        clean = stepper.advance_canonical(clean).state
    displaced = _band_perturbation(
        stepper,
        clean,
        PERTURBATION_SEEDS[index],
        settings.perturbation_bands[index],
    )
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    return {
        "index": index,
        "seed": seed,
        "initial": _state_metadata(stepper, initial),
        "post_burnin": _state_metadata(stepper, clean),
        "inputs": (
            {
                "case": f"clean_{index}",
                "family": "clean",
                "seed": seed,
                "state": clean,
                **_state_metadata(stepper, clean),
            },
            {
                "case": f"displaced_{index}",
                "family": "displaced",
                "seed": PERTURBATION_SEEDS[index],
                "band": list(settings.perturbation_bands[index]),
                "relative_perturbation_rms": float(
                    np.sqrt(np.mean((displaced - clean) ** 2)) / clean_rms
                ),
                "state": displaced,
                **_state_metadata(stepper, displaced),
            },
        ),
    }


def _build_inputs(
    base_config: KolmogorovReferenceConfig, settings: R1Settings
) -> list[dict[str, object]]:
    context = mp.get_context("spawn")
    outputs = []
    with ProcessPoolExecutor(
        max_workers=min(settings.workers, 3), mp_context=context
    ) as executor:
        futures = [
            executor.submit(_input_seed_worker, asdict(base_config), settings, index)
            for index in range(3)
        ]
        for future in as_completed(futures):
            outputs.append(future.result())
    outputs.sort(key=lambda row: int(row["index"]))
    return outputs


def verify_parent_input_replay(
    parent: dict[str, Any], outputs: list[dict[str, object]]
) -> dict[str, object]:
    """Verify regenerated initial, burn-in, and calibration-input hashes."""

    stationarity = parent["stationarity"]
    population = parent["population"]
    expected_initial = {
        int(row["seed"]): str(row["sha256"])
        for row in stationarity["initial_states"]
    }
    expected_burnin = {
        int(row["seed"]): str(row["sha256"])
        for row in stationarity["chosen_post_burnin_states"]
    }
    expected_inputs = {
        str(row["case"]): str(row["sha256"])
        for row in population["calibration_inputs"]
    }
    rows = []
    for output in outputs:
        seed = int(output["seed"])
        input_matches = {
            str(item["case"]): str(item["sha256"])
            == expected_inputs[str(item["case"])]
            for item in output["inputs"]
        }
        rows.append(
            {
                "seed": seed,
                "initial_hash_match": output["initial"]["sha256"]
                == expected_initial[seed],
                "post_burnin_hash_match": output["post_burnin"]["sha256"]
                == expected_burnin[seed],
                "input_hash_matches": input_matches,
            }
        )
    passed = all(
        row["initial_hash_match"]
        and row["post_burnin_hash_match"]
        and all(row["input_hash_matches"].values())
        for row in rows
    )
    if not passed:
        raise RuntimeError("R2 regenerated inputs do not match parent Q1")
    return {"rows": rows, "pass": True}


def compare_shared_spatial_rows(
    current_rows: list[dict[str, object]],
    retained_rows: list[dict[str, object]],
    *,
    tolerance: float = SPATIAL_REPLAY_ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    """Compare the complete N128-to-N256 overlap against retained R1 rows."""

    pair = "128_to_256"
    key_fields = ("case", "family", "pair", "low_resolution", "high_resolution", "horizon")
    numeric_fields = (
        "state_relative_l2",
        "energy_relative_difference",
        "enstrophy_relative_difference",
        "palinstrophy_relative_difference",
        "spectrum_total_variation",
    )

    def indexed(rows: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
        return {
            tuple(row[field] for field in key_fields): row
            for row in rows
            if row["pair"] == pair
        }

    current = indexed(current_rows)
    retained = indexed(retained_rows)
    keys_match = current.keys() == retained.keys() and bool(current)
    categorical_match = keys_match and all(
        bool(current[key]["finite"]) == bool(retained[key]["finite"])
        for key in current
    )
    differences = [
        abs(float(current[key][field]) - float(retained[key][field]))
        for key in current.keys() & retained.keys()
        for field in numeric_fields
    ]
    maximum_difference = max(differences, default=float("inf"))
    passed = keys_match and categorical_match and maximum_difference <= tolerance
    return {
        "pair": pair,
        "row_count": len(current),
        "keys_match": keys_match,
        "categorical_match": categorical_match,
        "maximum_absolute_numeric_difference": maximum_difference,
        "absolute_tolerance": tolerance,
        "pass": passed,
    }


def summarize_temporal_rows(
    rows: list[dict[str, object]],
    *,
    candidate_dt_max: float,
    reference_dt_max: float,
    maximum_projection_relative_l2: float,
    all_finite: bool,
    canonical_tolerance: float,
) -> dict[str, object]:
    """Apply the unchanged Q1 time-refinement gates to combined R2 rows."""

    selected = [row for row in rows if row["dt_max"] == candidate_dt_max]
    families = {}
    for family in ("clean", "displaced"):
        family_rows = [row for row in selected if row["family"] == family]
        if not family_rows:
            raise ValueError(f"missing temporal rows for family {family}")
        one_step = np.asarray(
            [row["one_step_relative_l2"] for row in family_rows], dtype=np.float64
        )
        path = np.asarray(
            [row["path_relative_l2"] for row in family_rows], dtype=np.float64
        )
        families[family] = {
            "one_step_median": float(np.median(one_step)),
            "one_step_maximum": float(np.max(one_step)),
            "path_median": float(np.median(path)),
            "path_maximum": float(np.max(path)),
        }
    one_step_pass = all(
        row["one_step_median"] < 1.0e-5 and row["one_step_maximum"] < 1.0e-4
        for row in families.values()
    )
    path_pass = all(
        row["path_median"] < 1.0e-3 and row["path_maximum"] < 5.0e-3
        for row in families.values()
    )
    structure_pass = all(
        float(row[metric]) < 0.02
        for row in selected
        for metric in (
            "energy_wasserstein",
            "enstrophy_wasserstein",
            "spectrum_total_variation",
        )
    )
    closure_pass = maximum_projection_relative_l2 <= canonical_tolerance
    passed = one_step_pass and path_pass and structure_pass and all_finite and closure_pass
    return {
        "candidate_dt_max": candidate_dt_max,
        "reference_dt_max": reference_dt_max,
        "families": families,
        "one_step_pass": one_step_pass,
        "path_pass": path_pass,
        "structure_pass": structure_pass,
        "all_finite": all_finite,
        "maximum_projection_relative_l2": maximum_projection_relative_l2,
        "closure_pass": closure_pass,
        "pass": passed,
    }


def _temporal_pair_worker(
    config_values: dict[str, object],
    settings: Q1Settings,
    pair: list[dict[str, object]],
) -> dict[str, object]:
    config = KolmogorovReferenceConfig(**config_values).validated()
    diagnostic, _, seconds = _time_refinement(config, settings, pair)
    return {"diagnostic": diagnostic, "seconds": seconds}


def _run_temporal(
    base_config: KolmogorovReferenceConfig,
    spatial_settings: R1Settings,
    temporal_settings: Q1Settings,
    inputs: list[dict[str, object]],
) -> tuple[dict[str, object], float]:
    started = perf_counter()
    candidate_resolution = spatial_settings.spatial_resolutions[1]
    candidate_config = replace(base_config, resolution=candidate_resolution)
    pairs = []
    for index in range(3):
        pair = []
        for item in inputs[index]["inputs"]:
            state = item["state"]
            if not isinstance(state, np.ndarray):
                raise TypeError("recreated input state is not an array")
            pair.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"sha256", "rms", "projection_relative_l2"}
                }
                | {"state": resize_dealiased_vorticity(state, candidate_resolution)}
            )
        pairs.append(pair)

    context = mp.get_context("spawn")
    outputs = []
    with ProcessPoolExecutor(
        max_workers=TEMPORAL_WORKERS, mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _temporal_pair_worker,
                asdict(candidate_config),
                temporal_settings,
                pair,
            )
            for pair in pairs
        ]
        for future in as_completed(futures):
            outputs.append(future.result())
    rows = [
        row
        for output in outputs
        for row in output["diagnostic"]["rows"]
    ]
    child_summaries = [output["diagnostic"]["summary"] for output in outputs]
    candidate_dt_max = max(temporal_settings.time_steps)
    reference_dt_max = min(temporal_settings.time_steps)
    summary = summarize_temporal_rows(
        rows,
        candidate_dt_max=candidate_dt_max,
        reference_dt_max=reference_dt_max,
        maximum_projection_relative_l2=max(
            float(row["maximum_projection_relative_l2"])
            for row in child_summaries
        ),
        all_finite=all(bool(row["all_finite"]) for row in child_summaries),
        canonical_tolerance=base_config.canonical_tolerance,
    )
    first_clean = pairs[0][0]["state"]
    if not isinstance(first_clean, np.ndarray):
        raise TypeError("first temporal state is not an array")
    repeatability, repeatability_seconds = _process_repeatability(
        replace(candidate_config, dt_max=candidate_dt_max), first_clean
    )
    summary["rows"] = sorted(
        rows, key=lambda row: (str(row["case"]), float(row["dt_max"]))
    )
    summary["repeatability"] = repeatability
    summary["worker_seconds"] = [float(output["seconds"]) for output in outputs]
    summary["repeatability_seconds"] = repeatability_seconds
    summary["pass_with_repeatability"] = bool(summary["pass"]) and bool(
        repeatability["pass"]
    )
    return summary, perf_counter() - started


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def verify_current_spatial_parent_replay(
    parent: dict[str, object], diagnostic: dict[str, object]
) -> dict[str, object]:
    """Verify this attempt's regenerated spatial inputs against parent Q1."""

    initial_rows = diagnostic["initial_rows"]
    post_burnin_rows = diagnostic["post_burnin_rows"]
    input_rows = diagnostic["input_rows"]
    outputs = []
    for row in initial_rows:
        seed = int(row["seed"])
        index = INITIAL_SEEDS.index(seed)
        outputs.append(
            {
                "seed": seed,
                "initial": row,
                "post_burnin": next(
                    item for item in post_burnin_rows if int(item["seed"]) == seed
                ),
                "inputs": [
                    item
                    for item in input_rows
                    if str(item["case"]).endswith(str(index))
                ],
            }
        )
    return _verify_spatial_parent_replay(parent, outputs)


def _write_packet(
    output_dir: Path,
    result: dict[str, object],
    source_hashes: dict[str, str],
    source_commit: str,
    parent_bindings: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": result["run_id"],
        "source_commit": source_commit,
        "parents": parent_bindings,
        "artifacts": {"result.json": _sha256(result_path)},
        "sources": source_hashes,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run reduced synthetic plumbing, not the registered R2 result",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _ensure_empty_output_dir(args.output_dir)
    parent, parent_binding = _verify_parent_packet()
    r1_results, r1_bindings = _verify_r1_packets()
    binding_start = _verify_source_binding(args.source_commit, quick=args.quick)
    hashes_start = _source_hashes()
    base_config, spatial_settings, _ = (
        _quick_contract() if args.quick else _full_contract()
    )
    total_started = perf_counter()

    spatial, spatial_rows, spatial_seconds = _run_spatial(base_config, spatial_settings)
    spatial["rows"] = spatial_rows
    if args.quick:
        parent_replay: dict[str, object] = {
            "status": "not_applicable_in_quick_mode",
            "pass": True,
        }
        shared_replay: dict[str, object] = {
            "status": "not_applicable_in_quick_mode",
            "pass": True,
        }
    else:
        parent_replay = verify_current_spatial_parent_replay(parent, spatial)
        shared_replay = compare_shared_spatial_rows(
            spatial_rows,
            r1_results["spatial"]["spatial_diagnostic"]["rows"],
        )
    if args.quick:
        passed = (
            bool(spatial["all_finite"])
            and bool(spatial["closure_pass"])
            and bool(spatial["repeatability"]["pass"])
        )
    else:
        passed = (
            bool(spatial["screen_pass"])
            and bool(shared_replay["pass"])
            and bool(parent_replay["pass"])
        )
    classification = (
        "quick_plumbing_pass" if passed else "quick_plumbing_fail"
    ) if args.quick else (
        "spatial_candidate_qualified" if passed else "spatial_candidate_failed"
    )
    result: dict[str, object] = {
        "run_id": f"{RUN_ID}-QUICK" if args.quick else RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_quick_spatial"
        if args.quick
        else "registered_solver_only_r2_b_spatial",
        "classification": classification,
        "source_commit": args.source_commit,
        "source_binding": {},
        "parent_bindings": {"q1": parent_binding, **r1_bindings},
        "parent_state_replay": parent_replay,
        "reference_config": asdict(base_config),
        "spatial_settings": asdict(spatial_settings),
        "spatial_diagnostic": spatial,
        "shared_r1_spatial_replay": shared_replay,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "multiprocessing_start_method": "spawn",
            "spatial_workers": spatial_settings.workers,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "timing_seconds": {
            "spatial": spatial_seconds,
            "total": perf_counter() - total_started,
        },
        "data_access": False,
        "checkpoint_access": False,
        "model_access": False,
        "training": False,
        "remote_execution": False,
        "test_access": False,
        "full_state_trajectory_retained": False,
    }
    binding_end = _verify_source_binding(args.source_commit, quick=args.quick)
    hashes_end = _source_hashes()
    if hashes_start != hashes_end:
        raise RuntimeError("registered R2 source hashes changed during execution")
    result["source_binding"] = {
        "start": binding_start,
        "end": binding_end,
        "hashes_stable_during_execution": True,
    }
    manifest = _write_packet(
        args.output_dir,
        result,
        hashes_end,
        args.source_commit,
        {"q1": parent_binding, **r1_bindings},
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "classification": classification,
                "total_seconds": result["timing_seconds"]["total"],
                "result_sha256": manifest["artifacts"]["result.json"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

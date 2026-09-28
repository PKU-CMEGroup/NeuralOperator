"""Evaluation-only rollout of the immutable 49,152-update Clean pilot.

Run ``python -m scripts.time_dependent_no.evaluate_kolmogorov_clean_rollout
--parent <C-packet> --parent-source <frozen-C-source-root>
--clean <Clean-packet> --clean-source <frozen-Clean-source-root>
--output <fresh> --device cuda``. All eight training and four existing validation
initial conditions are evaluated. No training, solver queries or correctors.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_clean as clean

RUN_ID = "CM_NEXT_KF_CLEAN_ROLLOUT_20260908A"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_MANIFEST_SHA256 = (
    "a6ffc5512a3431095dbc588acf2f86b0b070ef647cd3ddccf8e8e2739d83fcd4"
)
CHECKPOINT_SHA256 = "56469747abafdfd34dbedb3cf1c843a54f5ec8dfeaf1feda383756060eeec454"
SOURCE_PATHS = (
    "scripts/time_dependent_no/evaluate_kolmogorov_clean_rollout.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_clean_rollout.py",
    *clean.SOURCE_PATHS,
)
SNAPSHOT_STEPS = (0, 1, 8, 32, 64, 128, 256, 512)
HORIZONS = (1, 32, 128, 512)
MAXIMUM_RMS_OVER_TRAIN_SCALE = 1e6
WALL_SECONDS = 1800


def validate_clean(packet, source, *, unit_fixture=False):
    packet, source = Path(packet), Path(source)
    digest = clean._hash(packet / "artifact_manifest.json")
    if not unit_fixture and digest != CLEAN_MANIFEST_SHA256:
        raise ValueError("Clean packet differs from the pinned completed pilot")
    manifest = clean._read(packet / "artifact_manifest.json")
    expected = clean.CLEAN_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if (
        manifest["run_id"] != expected
        or manifest["source_stable"] is not True
        or set(manifest["sources"]) != set(clean.SOURCE_PATHS)
    ):
        raise ValueError("Clean identity/source closure mismatch")
    if {p.name for p in packet.iterdir()} != set(manifest["artifacts"]) | {
        "artifact_manifest.json"
    }:
        raise ValueError("Clean artifact inventory mismatch")
    for root, values in (
        (packet, manifest["artifacts"]),
        (source, manifest["sources"]),
        (REPO_ROOT, manifest["sources"]),
    ):
        if any(
            clean._hash(clean._bound(root, name)) != value
            for name, value in values.items()
        ):
            raise ValueError("Clean artifact or archived/live source hash mismatch")
    result = clean._read(packet / "result.json")
    if (
        result["run_id"] != expected
        or result["status"] != "completed"
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or result["parent_evidence_stable"] is not True
        or result["rollouts_evaluated"] is not False
        or result["development_used_for_updates"] is not False
        or result["protected_access"] is not False
    ):
        raise ValueError("Clean completion or access contract mismatch")
    selected = (
        f"terminal_{result['updates_completed']:06d}.pt"
        if unit_fixture
        else "terminal_049152.pt"
    )
    if not unit_fixture and (
        result["accepted_checkpoint"] != selected
        or manifest["artifacts"][selected] != CHECKPOINT_SHA256
    ):
        raise ValueError("only the accepted immutable terminal may be evaluated")
    terminal = result["terminal_checkpoints"][str(result["updates_completed"])]
    evaluation = result["evaluations"][-1]
    if (
        terminal["file"] != selected
        or terminal["sha256"] != manifest["artifacts"][selected]
        or evaluation["sha256"] != manifest["artifacts"][evaluation["file"]]
        or evaluation["update"] != result["updates_completed"]
    ):
        raise ValueError("checkpoint/teacher evaluation identity mismatch")
    return result, {
        "manifest_sha256": digest,
        "checkpoint": selected,
        "checkpoint_sha256": manifest["artifacts"][selected],
        "teacher_file": evaluation["file"],
        "artifacts": {**manifest["artifacts"], "artifact_manifest.json": digest},
        "sources": manifest["sources"],
    }


def field_diagnostics(state):
    """Physical diagnostics on actual values, without silently projecting them."""
    state = np.asarray(state, dtype=np.float64)
    if (
        state.ndim != 2
        or state.shape[0] != state.shape[1]
        or not np.isfinite(state).all()
    ):
        raise ValueError("diagnostic state must be finite and square")
    n = state.shape[0]
    k = np.fft.fftfreq(n) * n
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    power = np.abs(np.fft.fft2(state) / state.size) ** 2
    inverse = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 > 0)
    return {
        "mean_vorticity": float(np.mean(state)),
        "kinetic_energy": float(0.5 * np.sum(power * inverse)),
        "enstrophy": float(0.5 * np.mean(state**2)),
        "palinstrophy": float(0.5 * np.sum(power * k2)),
    }


def spectral_error_sse(error, modes):
    """Exact SSE partition by rectangular Fourier bands; not tangent/normal."""
    n = error.shape[0]
    k = np.abs(np.fft.fftfreq(n) * n)
    radius = np.maximum(k[:, None], k[None, :])
    power = np.abs(np.fft.fft2(error)) ** 2 / error.size
    return np.array(
        [
            power[radius <= modes].sum(),
            power[(radius > modes) & (radius <= n // 3)].sum(),
            power[radius > n // 3].sum(),
        ],
        dtype=np.float64,
    )


def rollout_one(model, truth, train_scale, modes, deadline, *, progress=None):
    """Feedback is next_state only. Stop failed paths individually; never reset."""
    device = next(model.parameters()).device
    truth = np.asarray(truth)
    if truth.dtype != np.float32 or truth.ndim != 3 or not np.isfinite(truth).all():
        raise ValueError("truth must be the finite float32 native state store")
    if not np.isfinite(train_scale) or train_scale <= 0:
        raise ValueError("invalid fixed training scale")
    steps, n = truth.shape[0] - 1, truth.shape[-1]
    snapshots = sorted(set(t for t in SNAPSHOT_STEPS if t <= steps) | {steps})
    current = torch.from_numpy(truth[0].copy()).unsqueeze(0).to(device)
    rows = []
    retained = {
        "step": [0],
        "previous_input": [truth[0].copy()],
        "raw_output": [truth[0].copy()],
        "next_state": [truth[0].copy()],
        "truth": [truth[0].copy()],
    }
    status, error_message = "completed", None
    model.eval()
    with torch.no_grad():
        for t in range(1, steps + 1):
            if perf_counter() >= deadline:
                status = "incomplete_budget"
                break
            previous = current[0].cpu().numpy().copy()
            try:
                prediction = model(current)
                raw = prediction["raw_next"][0].cpu().numpy().copy()
                nxt = prediction["next_state"][0].cpu().numpy().copy()
                if (
                    raw.shape != (n, n)
                    or nxt.shape != (n, n)
                    or raw.dtype != np.float32
                    or nxt.dtype != np.float32
                ):
                    raise RuntimeError("model changed the deployed state shape/dtype")
                if not np.isfinite(raw).all() or not np.isfinite(nxt).all():
                    status = "nonfinite_prediction"
                    break
                current = prediction["next_state"]
                r, z, y = (a.astype(np.float64) for a in (raw, nxt, truth[t]))
                difference = z - y
                row = {
                    "step": t,
                    "rollout_sse": float(np.sum(difference**2)),
                    "raw_rollout_sse": float(np.sum((r - y) ** 2)),
                    "target_sse": float(np.sum(y**2)),
                    "persistence_rollout_sse": float(
                        np.sum((truth[0].astype(np.float64) - y) ** 2)
                    ),
                    "restriction_sse": float(np.sum((r - z) ** 2)),
                    "prediction_rms_over_train_scale": float(
                        np.sqrt(np.mean(z**2)) / train_scale
                    ),
                    "spectral_error_sse": spectral_error_sse(
                        difference, modes
                    ).tolist(),
                    "prediction_structure": field_diagnostics(z),
                    "truth_structure": field_diagnostics(y),
                }
                rows.append(row)
                if (
                    t in snapshots
                    or row["prediction_rms_over_train_scale"]
                    > MAXIMUM_RMS_OVER_TRAIN_SCALE
                ):
                    for key, value in (
                        ("step", t),
                        ("previous_input", previous),
                        ("raw_output", raw),
                        ("next_state", nxt),
                        ("truth", truth[t]),
                    ):
                        retained[key].append(
                            value.copy() if isinstance(value, np.ndarray) else value
                        )
                if progress is not None:
                    progress(row)
                if (
                    row["prediction_rms_over_train_scale"]
                    > MAXIMUM_RMS_OVER_TRAIN_SCALE
                ):
                    status = "amplitude_limit"
                    break
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                OverflowError,
            ) as error:
                status = (
                    "nonfinite_prediction"
                    if isinstance(error, ValueError)
                    and str(error) == "vorticity must contain only finite values"
                    else "model_error"
                )
                error_message = str(error)
                break
    return {
        "status": status,
        "completed_steps": len(rows),
        "failed_at_step": None
        if status == "completed"
        else len(rows)
        if status == "amplitude_limit"
        else len(rows) + 1,
        "error": error_message,
        "rows": rows,
    }, {k: np.asarray(v) for k, v in retained.items()}


def summarize(case, teacher_sse, train_scale, node_count, horizons):
    summaries = []
    for horizon in horizons:
        if case["completed_steps"] < horizon or (
            case["failed_at_step"] is not None and case["failed_at_step"] <= horizon
        ):
            summaries.append(
                {"horizon": horizon, "complete": False, "rollout_relative_l2": None}
            )
            continue
        rows = case["rows"][:horizon]
        sse = sum(r["rollout_sse"] for r in rows)
        target = sum(r["target_sse"] for r in rows)
        per_step = [
            np.sqrt(r["rollout_sse"] / r["target_sse"])
            for r in rows
            if r["target_sse"] > 0
        ]
        summaries.append(
            {
                "horizon": horizon,
                "complete": True,
                "rollout_relative_l2": float(np.sqrt(sse / target))
                if target > 0
                else None,
                "teacher_relative_l2": float(
                    np.sqrt(float(np.sum(teacher_sse[:horizon])) / target)
                )
                if target > 0
                else None,
                "mean_step_relative_l2": float(np.mean(per_step))
                if len(per_step) == horizon
                else None,
                "train_scale_rmse": float(
                    np.sqrt(sse / (horizon * node_count)) / train_scale
                ),
                "persistence_rollout_relative_l2": float(
                    np.sqrt(sum(r["persistence_rollout_sse"] for r in rows) / target)
                )
                if target > 0
                else None,
            }
        )
    return summaries


def run(
    parent, parent_source, packet, source, output, device_name, *, unit_fixture=False
):
    if device_name != ("cpu" if unit_fixture else "cuda"):
        raise ValueError(
            "real evaluation uses the selected CUDA resource; CPU is fixture-only"
        )
    parent, parent_source, packet, source, output = map(
        Path, (parent, parent_source, packet, source, output)
    )
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    sources = {name: clean._hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    previous, evidence = validate_clean(packet, source, unit_fixture=unit_fixture)
    receipt = {}
    population = clean.load_population(
        parent, parent_source, unit_fixture=unit_fixture, receipt=receipt
    )
    if (
        previous["data"]["state_store_sha256"]
        != population.metadata["state_store_sha256"]
        or previous["data"]["evidence"] != receipt["evidence"]
        or previous["data"]["train_input_rms_float64"] != population.train_scale
    ):
        raise ValueError("rollout population or normalization differs from training")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = clean.PeriodicVorticityPCNO(
        **previous["model_config"], train_scale=population.train_scale
    )
    # The locally produced checkpoint was fully hash-verified before unpickling.
    checkpoint = torch.load(
        packet / evidence["checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint["identity"] != previous["checkpoint_identity"]
        or checkpoint["update"] != previous["updates_completed"]
        or checkpoint["schedule_position"] != checkpoint["update"]
    ):
        raise ValueError("checkpoint identity/schedule mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device_name).eval()
    replay = clean._terminal_replay(model, packet / evidence["teacher_file"])
    with np.load(packet / evidence["teacher_file"], allow_pickle=False) as saved:
        teacher = (
            saved["learned_sse"].reshape(12, population.states.shape[1] - 1).copy()
        )
    output.mkdir()
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": sources,
        "clean_evidence": evidence,
        "population_evidence": receipt["evidence"],
        "checkpoint_replay": replay,
        "train_scale": float(model.train_scale),
        "horizon": population.states.shape[1] - 1,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "amplitude_limit_rms_over_train_scale": MAXIMUM_RMS_OVER_TRAIN_SCALE,
        "wall_seconds": WALL_SECONDS,
        "cases": [],
        "optimization_steps": 0,
        "solver_calls": 0,
        "protected_access": False,
        "network_access": False,
        "geometry_fitted": False,
        "corrections_added": False,
        "interpretation": (
            "Exploratory pilot diagnosis with an existing train/validation one-step "
            "gap. Training starts test recurrence on supervised paths; validation "
            "results remain confounded. No claim of manifold drift, correction "
            "necessity or prospective confirmation. Shared mean-zero 2/3 restriction "
            "is unchanged; raw output is never recurrent feedback. Energy/enstrophy "
            "are diagnostics, not conserved quantities of this forced dissipative "
            "system."
        ),
    }
    clean._write(output / "launch.json", record)
    for i, (seed, role) in enumerate(clean.SEED_ROLES):
        with (output / f"progress_{seed}.jsonl").open("x", encoding="utf-8") as log:

            def progress(row):
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()

            case, snapshots = rollout_one(
                model,
                population.states[i],
                float(model.train_scale),
                previous["model_config"]["modes"],
                started + WALL_SECONDS,
                progress=progress,
            )
        horizons = (1, 2, 4) if unit_fixture else HORIZONS
        case.update(seed=seed, role=role, teacher_sse=teacher[i].tolist())
        case["summaries"] = summarize(
            case,
            teacher[i],
            float(model.train_scale),
            population.states.shape[-1] ** 2,
            horizons,
        )
        snapshots_path = output / f"snapshots_{seed}.npz"
        with snapshots_path.open("xb") as stream:
            np.savez_compressed(stream, **snapshots)
        case.update(
            snapshot_file=snapshots_path.name,
            snapshot_sha256=clean._hash(snapshots_path),
        )
        clean._write(output / f"case_{seed}.json", case)
        record["cases"].append(case)
        clean._write(output / "progress.json", record)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "role": role,
                    "status": case["status"],
                    "completed_steps": case["completed_steps"],
                }
            ),
            flush=True,
        )
    after = {name: clean._hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    stable = all(
        clean._hash(clean._bound(root, name)) == value
        for root, hashes in (
            (packet, evidence["artifacts"]),
            (source, evidence["sources"]),
            (parent, receipt["evidence"]["artifacts"]),
            (parent_source, receipt["evidence"]["sources"]),
        )
        for name, value in hashes.items()
    )
    record.update(
        status="completed" if stable and after == sources else "invalid_provenance",
        sources_after=after,
        source_stable=after == sources,
        parent_evidence_stable=stable,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        all_rollouts_completed=all(c["status"] == "completed" for c in record["cases"]),
        failure_counts={
            role: sum(
                c["status"] != "completed" for c in record["cases"] if c["role"] == role
            )
            for role in ("train", "development")
        },
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device_name,
        },
    )
    clean._write(output / "result.json", record)
    clean._write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "status": record["status"],
            "sources": sources,
            "source_stable": record["source_stable"],
            "clean_manifest_sha256": evidence["manifest_sha256"],
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-source", "clean", "clean-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    args = parser.parse_args()
    result = run(
        args.parent,
        args.parent_source,
        args.clean,
        args.clean_source,
        args.output,
        args.device,
    )
    if result["status"] != "completed" or any(
        c["status"] in ("incomplete_budget", "model_error") for c in result["cases"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

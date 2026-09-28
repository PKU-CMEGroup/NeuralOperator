"""Train-only, offline reference-conditioned tangent forecast diagnostic.

Run with ``python -m scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast
--input PACKET --output NEW_DIRECTORY --device cpu``. Future clean states are
inputs: this is not a deployable forecast or a trusted physical-response assay.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO,
    canonicalize_vorticity,
)
from utility.time_dependent_no.forced_tangent import (
    decompose_forecast_error,
    finite_amplitude_response,
    forced_tangent_forecast,
)

ROOT = Path(__file__).resolve().parents[2]
REPLAY_LIMIT = 1e-6
FD_LIMIT = 1e-2
INPUT_FILES = {"reference.npz", "checkpoint.pt", "fit_result.json", "teacher.npz"}
TRANSITION_STEPS = (4, 5, 6)
TRANSITION_AMPLITUDES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MODEL_SOURCES = {
    "utility/time_dependent_no/pcno_kolmogorov.py", "pcno/pcno.py",
    "pcno/geo_utility.py", "pcno/__init__.py", "utility/__init__.py",
    "utility/time_dependent_no/__init__.py", "utility/adam.py",
    "utility/losses.py", "utility/normalizer.py",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in arrays.items()
        })
    return sha256(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(path):
    """Verify all captured bytes before any array/checkpoint deserialization."""
    path = Path(path).resolve()
    manifest_bytes = (path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = dict(schema_version=1, role="train", start_step=0)
    if (any(manifest.get(k) != v for k, v in expected.items())
            or manifest.get("seed") not in range(2026090611, 2026090619)
            or manifest.get("model_name") not in ("clean8", "clean32")
            or type(manifest.get("horizon")) is not int
            or not 1 <= manifest["horizon"] <= 32):
        raise ValueError("input scope must use an original train path and horizon 1:32")
    if set(manifest["artifacts"]) != INPUT_FILES:
        raise ValueError("unexpected input artifact set")
    if not MODEL_SOURCES <= set(manifest["sources"]):
        raise ValueError("incomplete frozen model source closure")
    for name, digest in manifest["sources"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or source.suffix != ".py":
            raise ValueError("invalid model source path")
        if sha256(source) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    captured = {name: (path / name).read_bytes() for name in sorted(INPUT_FILES)}
    for name, value in captured.items():
        if hashlib.sha256(value).hexdigest() != manifest["artifacts"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    fit = json.loads(captured["fit_result.json"])
    if manifest["model_name"] == "clean32":
        terminal = fit["terminal_checkpoint"]
        if terminal["update"] != 49152:
            raise ValueError("fit terminal update mismatch")
    else:
        terminal = fit["terminal_checkpoints"]["49152"]
        if fit["accepted_checkpoint"] != "terminal_049152.pt":
            raise ValueError("fit accepted checkpoint mismatch")
    if (terminal["sha256"] != manifest["artifacts"]["checkpoint.pt"]
            or fit["updates_completed"] != 49152
            or terminal["file"] != "terminal_049152.pt"
            or fit["model_config"] != fit["checkpoint_identity"]["model_config"]):
        raise ValueError("fit terminal checkpoint/configuration mismatch")
    with np.load(io.BytesIO(captured["reference.npz"]), allow_pickle=False) as saved:
        reference = saved["states"].copy()
    n = fit["model_config"]["resolution"]
    if reference.shape != (manifest["horizon"] + 1, n, n) or reference.dtype != np.float32:
        raise ValueError("reference must be float32 [horizon+1,n,n]")
    if not np.isfinite(reference).all():
        raise ValueError("nonfinite reference")
    return manifest, captured, fit, torch.from_numpy(reference), hashlib.sha256(manifest_bytes).hexdigest()


def load_model(captured, fit, device):
    checkpoint = torch.load(io.BytesIO(captured["checkpoint.pt"]),
                            map_location="cpu", weights_only=False)
    identity = fit["checkpoint_identity"]
    if (checkpoint["identity"] != identity or checkpoint["update"] != 49152
            or checkpoint["schedule_position"] != 49152):
        raise ValueError("terminal checkpoint identity/schedule mismatch")
    model = PeriodicVorticityPCNO(**fit["model_config"],
                                train_scale=identity["train_scale_float64"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.train_scale) != identity["train_scale_model_float32"]:
        raise ValueError("checkpoint normalizer mismatch")
    model.to(device).eval().requires_grad_(False)
    return model


def terminal_replay(model, teacher_bytes, device, training_paths):
    with np.load(io.BytesIO(teacher_bytes), allow_pickle=False) as saved, torch.no_grad():
        indices = saved["sentinel_global_indices"]
        if (indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= training_paths * 512)):
            raise ValueError("sentinel replay requires training-only global indices")
        if any(saved[key].shape[0] != len(indices) for key in
               ("sentinel_input", "sentinel_raw", "sentinel_next")):
            raise ValueError("sentinel row count mismatch")
        output = model(torch.from_numpy(saved["sentinel_input"]).to(device))
        errors = {}
        for name, key in (("raw", "raw_next"), ("next", "next_state")):
            actual = output[key].cpu().numpy().astype(np.float64)
            expected = saved["sentinel_" + name].astype(np.float64)
            numerator = np.sqrt(np.mean((actual - expected) ** 2, axis=(-2, -1)))
            denominator = np.sqrt(np.mean(expected**2, axis=(-2, -1)))
            relative = np.divide(numerator, denominator,
                                 out=np.full_like(numerator, np.inf), where=denominator > 0)
            relative[(denominator == 0) & (numerator == 0)] = 0
            if not np.isfinite(relative).all() or np.any(relative > REPLAY_LIMIT):
                raise RuntimeError("terminal checkpoint sentinel replay failed")
            errors[name] = relative.tolist()
    return {"relative_rms_limit": REPLAY_LIMIT, "errors": errors}


def rms(value):
    return float(value.detach().double().square().mean().sqrt().cpu())


def cosine(left, right):
    left, right = left.detach().double(), right.detach().double()
    denominator = float(left.norm() * right.norm())
    return float((left * right).sum()) / denominator if denominator else None


def validate_jvp(step, state, scale):
    n = state.shape[-1]
    coordinate = torch.arange(n, device=state.device) * (2 * math.pi / n)
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    direction = canonicalize_vorticity((torch.sin(x) + 0.5 * torch.cos(2 * y))[None])[0]
    direction = direction * (scale / rms(direction))
    _, derivative = torch.autograd.functional.jvp(lambda z: step(0, z), state, direction)
    rows = []
    for epsilon in (1e-2, 1e-3):
        with torch.no_grad():
            difference = (step(0, state + epsilon * direction)
                          - step(0, state - epsilon * direction)) / (2 * epsilon)
        error = rms(difference - derivative)
        denominator = rms(derivative)
        relative = error / denominator if denominator else (0.0 if not error else None)
        rows.append({"epsilon_train_scale_rms": epsilon, "relative_rms_error":
                     relative if relative is not None and math.isfinite(relative) else None})
    return {"limit": FD_LIMIT, "rows": rows,
            "passed": any(r["relative_rms_error"] is not None
                          and r["relative_rms_error"] <= FD_LIMIT for r in rows)}


def physical(state):
    value = state.detach().double()
    return {"finite": bool(torch.isfinite(value).all()), "mean": float(value.mean()),
            "rms": rms(value), "enstrophy": float(value.square().mean() / 2),
            "maxabs": float(value.abs().max())}


def compare_forecast(forecast, observed_errors, scale):
    z, q, b = (forecast[k] for k in
               ("predicted_error", "identity_response_error", "clean_forcing"))
    rows = []
    for n, error in enumerate(observed_errors):
        row = {"step": n, "observed_rms_scaled": rms(error) / scale,
               "forecast_rms_scaled": rms(z[n]) / scale,
               "identity_response_rms_scaled": rms(q[n]) / scale,
               "forecast_mismatch_rms_scaled": rms(z[n] - error) / scale,
               "identity_mismatch_rms_scaled": rms(q[n] - error) / scale,
               "forecast_error_cosine": cosine(z[n], error)}
        if n:
            propagated = z[n] - b[n - 1]
            row.update(forcing_rms_scaled=rms(b[n - 1]) / scale,
                       propagated_rms_scaled=rms(propagated) / scale,
                       forcing_propagated_inner_product_scaled=float(
                           (b[n - 1].double() * propagated.double()).mean()) / scale**2,
                       forcing_propagated_cosine=cosine(b[n - 1], propagated))
        rows.append(row)
    end = len(observed_errors)
    numerator = (z[2:end].double() - observed_errors[2:].double()).square().sum().item()
    denominator = (q[2:end].double() - observed_errors[2:].double()).square().sum().item()
    return rows, None if denominator == 0 else 1 - numerator / denominator


def load_parent(path, expected_sha256, manifest, manifest_hash, reference):
    """Authenticate the completed parent before decoding its saved arrays."""
    path = Path(path).resolve()
    raw = (path / "result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("parent result hash mismatch")
    parent = json.loads(raw)
    if parent["status"] != "completed" or parent["input_manifest_sha256"] != manifest_hash:
        raise ValueError("parent status/input identity mismatch")
    if any(parent["sources"].get(name) != digest for name, digest in manifest["sources"].items()):
        raise ValueError("parent frozen model source mismatch")
    artifacts = parent["artifacts"]
    if not {"forecast.npz", "observed.npz", "forecast_freeze.json"} <= set(artifacts):
        raise ValueError("parent forecast/observation/freeze missing")
    captured = {}
    for name, digest in artifacts.items():
        member = (path / name).resolve()
        if not member.is_relative_to(path) or member == path:
            raise ValueError("invalid parent artifact path")
        value = member.read_bytes()
        if hashlib.sha256(value).hexdigest() != digest:
            raise ValueError(f"parent artifact hash mismatch: {name}")
        if name in ("forecast.npz", "observed.npz", "forecast_freeze.json"):
            captured[name] = value
    freeze = json.loads(captured["forecast_freeze.json"])
    if freeze["sha256"] != artifacts["forecast.npz"] or freeze["input_manifest_sha256"] != manifest_hash:
        raise ValueError("parent freeze identity mismatch")
    decoded = []
    for name in ("forecast.npz", "observed.npz"):
        with np.load(io.BytesIO(captured[name]), allow_pickle=False) as saved:
            decoded.append({key: torch.from_numpy(saved[key].copy()) for key in saved.files})
    forecast, observed = decoded
    for key in ("predicted_error", "identity_response_error", "clean_forcing"):
        expected = reference[:-1].shape if key == "clean_forcing" else reference.shape
        if forecast[key].shape != expected or forecast[key].dtype != reference.dtype or not torch.isfinite(forecast[key]).all():
            raise ValueError("parent forecast shape/dtype/finiteness mismatch")
    for key in ("states", "errors"):
        if observed[key].shape != reference.shape or observed[key].dtype != reference.dtype or not torch.isfinite(observed[key]).all():
            raise ValueError("parent observation shape/dtype/finiteness mismatch")
    if not torch.equal(observed["errors"], observed["states"] - reference):
        raise ValueError("parent saved error/state inconsistency")
    return parent, forecast, observed


def transition_probe(step, index, state, direction, scale, native_amplitudes=()):
    """Use the same absolute RMS amplitudes for either direction, then its native size."""
    native = rms(direction) / scale
    if not native:
        raise ValueError("zero direction has no transition response")
    amplitudes = list(TRANSITION_AMPLITUDES)
    for amplitude in (*native_amplitudes, native):
        if amplitude not in amplitudes:
            amplitudes.append(amplitude)
    probe = finite_amplitude_response(step, index, state, direction / native, amplitudes)
    with torch.no_grad():
        probe["base_no_grad"] = step(index, state).detach()
    offset = probe["base_value"].double() - probe["base_no_grad"].double()
    probe["base_kernel_offset"] = offset
    for key in ("even_response", "plus_remainder", "minus_remainder"):
        probe[key] = probe[key].double() + offset
    rows = []
    for i, amplitude in enumerate(amplitudes):
        linear, odd, even = (probe[key][i] for key in ("linear_response", "odd_response", "even_response"))
        plus, minus = probe["plus_displacement"][i], probe["minus_displacement"][i]
        linear_norm = rms(linear)
        ratio = lambda value: rms(value) / linear_norm if linear_norm else None
        rows.append({"amplitude_train_scale_rms": amplitude,
            "common_grid": amplitude in TRANSITION_AMPLITUDES, "native_amplitude": amplitude == native,
            **{key + "_rms_scaled": rms(probe[key][i]) / scale for key in
               ("linear_response", "odd_response", "even_response", "plus_remainder", "minus_remainder", "plus_displacement", "minus_displacement")},
            "plus_gain": rms(odd + even) / rms(plus), "minus_gain": rms(even - odd) / rms(minus),
            "odd_gain": rms(odd) / rms((plus - minus) / 2),
            "linear_gain": linear_norm / (amplitude * scale),
            "odd_minus_linear_rms_scaled": rms(odd.double() - linear.double()) / scale,
            "odd_minus_linear_over_linear": ratio(odd.double() - linear.double()),
            "plus_remainder_over_linear": ratio(probe["plus_remainder"][i]),
            "minus_remainder_over_linear": ratio(probe["minus_remainder"][i]),
            "even_over_linear": ratio(even), "linear_odd_cosine": cosine(linear, odd)})
    return probe, rows, native


def run_response_transition(input_path, output_path, parent_path, parent_sha256, device="cpu"):
    output = Path(output_path).resolve()
    for protected in (Path(input_path).resolve(), Path(parent_path).resolve()):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("output must be separate from immutable input and parent")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "incomplete", "phase": "inputs", "operation": "response_transition",
              "role": "posthoc_train_only", "model_forward_calls": 0, "solver_calls": 0,
              "optimization_steps": 0, "protected_access": False, "artifacts": {},
              "interpretation": "posthoc finite-amplitude diagnosis; actual-error directions are not forecasts; no certified radius",
              "precision": "model/JVP and utility signed-response differences are FP32; clean-base recentering and scalar summaries use FP64; near-roundoff remainders do not define a validity radius"}
    arrays = {}
    device = torch.device(device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        manifest, captured, fit, reference, manifest_hash = load_inputs(input_path)
        if (manifest["model_name"], manifest["seed"], manifest["horizon"]) != ("clean32", 2026090611, 32):
            raise ValueError("transition study is restricted to the original Clean32 train pilot")
        parent, forecast, observed = load_parent(parent_path, parent_sha256, manifest, manifest_hash, reference)
        sources = set(manifest["sources"]) | {"scripts/time_dependent_no/evaluate_kolmogorov_tangent_forecast.py", "utility/time_dependent_no/forced_tangent.py"}
        record.update(input_manifest_sha256=manifest_hash, parent_result_sha256=parent_sha256,
                      parent_artifacts=parent["artifacts"], sources={f: sha256(ROOT / f) for f in sorted(sources)},
                      common_amplitudes_train_scale_rms=TRANSITION_AMPLITUDES)
        record["phase"] = "checkpoint_replay"
        model = load_model(captured, fit, device)
        del captured["checkpoint.pt"]
        def count_forward(_model, _arguments):
            record["model_forward_calls"] += 1
        model.register_forward_pre_hook(count_forward)
        record["checkpoint_replay"] = terminal_replay(model, captured["teacher.npz"], device, 32)
        reference = reference.to(device)
        forecast = {k: v.to(device) for k, v in forecast.items()}
        observed = {k: v.to(device) for k, v in observed.items()}
        scale = float(model.train_scale)
        step = lambda n, state: model(state[None])["next_state"][0]
        record["phase"] = "jvp_validation"
        record["jvp_validation"] = validate_jvp(step, reference[0], scale)
        if not record["jvp_validation"]["passed"]:
            raise RuntimeError("JVP finite-difference numerical check failed")
        record["phase"] = "response_transition"
        record["responses"], record["decompositions"] = [], []
        for n in TRANSITION_STEPS:
            native_amplitudes = tuple(rms(value[n]) / scale for value in
                                      (forecast["predicted_error"], observed["errors"]))
            for label, direction in (("forecast", forecast["predicted_error"][n]), ("actual", observed["errors"][n])):
                probe, rows, native = transition_probe(step, n, reference[n], direction, scale, native_amplitudes)
                arrays.update({f"step_{n}_{label}_{key}": value for key, value in probe.items()})
                record["responses"].append({"step": n, "direction": label, "native_amplitude": native,
                    "clean_kernel_offset_rms_scaled": rms(probe["base_kernel_offset"]) / scale, "rows": rows})
            decomposition = decompose_forecast_error(step, n, reference, forecast, observed["errors"])
            arrays.update({f"step_{n}_decomposition_{key}": value for key, value in decomposition.items()})
            with torch.no_grad():
                replay = step(n, observed["states"][n])
            mismatch = replay - observed["states"][n + 1]
            arrays[f"step_{n}_actual_replay_mismatch"] = mismatch
            discrepancy = decomposition["discrepancy_next"].double()
            denominator = float(discrepancy.square().sum())
            projections = {key: (float((term.double() * discrepancy).sum()) / denominator if denominator else None)
                for key, term in (("propagation", decomposition["propagated_discrepancy"]),
                                  ("negative_remainder", -decomposition["actual_remainder"]),
                                  ("kernel_offset", decomposition["clean_kernel_offset"]),
                                  ("closure", decomposition["closure_residual"]))}
            record["decompositions"].append({"step": n,
                **{key + "_rms_scaled": rms(value) / scale for key, value in decomposition.items() if value.ndim},
                "closure_relative_l2": float(decomposition["closure_relative_l2"]),
                "replay_mismatch_rms_scaled": rms(mismatch) / scale,
                "replay_mismatch_relative_rms": rms(mismatch) / rms(observed["states"][n + 1]),
                "reconstructed_actual_input_mismatch_rms_scaled": rms(reference[n] + observed["errors"][n] - observed["states"][n]) / scale,
                "projection_fraction_onto_next_discrepancy": projections,
                "propagation_negative_remainder_cosine": cosine(decomposition["propagated_discrepancy"], -decomposition["actual_remainder"])})
        if sha256(Path(parent_path) / "result.json") != parent_sha256 or any(
                sha256(Path(parent_path) / name) != digest for name, digest in parent["artifacts"].items()):
            raise RuntimeError("immutable parent changed during transition study")
        record["parent_unchanged"] = True
        record["status"] = "completed"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if arrays:
            record["artifacts"]["response_transition.npz"] = save_arrays(output / "response_transition.npz", arrays)
        record["seconds"] = time.monotonic() - started
        record["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)}
        record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        record["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        write_json(output / "result.json", record)
    return record


def run(input_path, output_path, device="cpu"):
    output = Path(output_path).resolve()
    if output == Path(input_path).resolve() or output.is_relative_to(Path(input_path).resolve()):
        raise ValueError("output must be separate from immutable input")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "incomplete", "phase": "inputs", "model_forward_calls": 0,
              "solver_calls": 0, "optimization_steps": 0, "protected_access": False,
              "interpretation": "offline reference-conditioned diagnostic; not deployable or prospective",
              "artifacts": {}, "phases": []}
    observed, probe_arrays = [], {}
    device = torch.device(device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        manifest, captured, fit, reference_cpu, manifest_hash = load_inputs(input_path)
        sources = set(manifest["sources"]) | {
            str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "utility/time_dependent_no/forced_tangent.py"}
        record.update(input_manifest_sha256=manifest_hash, input_scope=manifest,
                      sources={name: sha256(ROOT / name) for name in sorted(sources)})
        record["phase"] = "checkpoint_replay"
        model = load_model(captured, fit, device)
        del captured["checkpoint.pt"]

        def count_forward(_model, _arguments):
            record["model_forward_calls"] += 1

        model.register_forward_pre_hook(count_forward)
        record["checkpoint_replay"] = terminal_replay(
            model, captured["teacher.npz"], device, 32 if manifest["model_name"] == "clean32" else 8)
        reference = reference_cpu.to(device)
        scale = float(model.train_scale)
        step = lambda n, state: model(state[None])["next_state"][0]
        record["phase"] = "jvp_validation"
        record["jvp_validation"] = validate_jvp(step, reference[0], scale)
        if not record["jvp_validation"]["passed"]:
            raise RuntimeError("JVP finite-difference numerical check failed")
        record["phase"] = "forecast"
        forecast = forced_tangent_forecast(step, reference)
        record["artifacts"]["forecast.npz"] = save_arrays(output / "forecast.npz", forecast)
        record["phases"].append({"phase": "forecast_frozen", "seconds": time.monotonic() - started,
                                 "sha256": record["artifacts"]["forecast.npz"],
                                 "input_manifest_sha256": manifest_hash,
                                 "sources": record["sources"],
                                 "interpretation": record["interpretation"],
                                 "forward_calls": record["model_forward_calls"]})
        write_json(output / "forecast_freeze.json", record["phases"][-1])
        record["artifacts"]["forecast_freeze.json"] = sha256(output / "forecast_freeze.json")
        record["phase"] = "nonlinear_rollout"
        state = reference[0].clone()
        observed.append(state.detach().cpu())
        with torch.no_grad():
            for n in range(len(reference) - 1):
                state = step(n, state)
                observed.append(state.detach().cpu())
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f"nonfinite nonlinear state at step {n + 1}")
        record["phases"].append({"phase": "rollout_completed", "seconds": time.monotonic() - started})
        errors = torch.stack(observed).to(device) - reference
        record["per_step"], record["skill_vs_identity_excluding_step_1"] = compare_forecast(forecast, errors, scale)
        record["physical"] = [{"step": n, "reference": physical(reference_cpu[n]),
                               "observed": physical(value)} for n, value in enumerate(observed)]
        record["phase"] = "finite_amplitude_probes"
        record["probes"] = []
        for n in (7, 15, 31):
            if n >= len(reference) - 1:
                continue
            direction = forecast["predicted_error"][n]
            if not rms(direction):
                record["probes"].append({"step": n, "status": "zero_forecast_displacement"})
                continue
            probe = finite_amplitude_response(step, n, reference[n], direction, (0.5, 1.0))
            probe_arrays.update({f"step_{n}_{k}": v for k, v in probe.items()})
            for index, multiplier in enumerate((0.5, 1.0)):
                record["probes"].append({"step": n, "multiplier": multiplier,
                    **{key + "_rms_scaled": rms(probe[key][index]) / scale for key in
                       ("linear_response", "odd_response", "even_response", "plus_remainder", "minus_remainder", "plus_displacement", "minus_displacement")},
                    "linear_odd_cosine": cosine(probe["linear_response"][index], probe["odd_response"][index])})
        if sha256(output / "forecast.npz") != record["artifacts"]["forecast.npz"]:
            raise RuntimeError("frozen forecast changed")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if observed:
            values = torch.stack(observed)
            record["artifacts"]["observed.npz"] = save_arrays(output / "observed.npz",
                {"states": values, "errors": values - reference_cpu[:len(values)]})
            record["observed_steps_retained"] = len(values) - 1
        if probe_arrays:
            record["artifacts"]["probes.npz"] = save_arrays(output / "probes.npz", probe_arrays)
        record["seconds"] = time.monotonic() - started
        record["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)}
        record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        record["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        write_json(output / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("tangent_forecast", "response_transition"), default="tangent_forecast")
    parser.add_argument("--parent-packet", type=Path)
    parser.add_argument("--parent-sha256")
    args = parser.parse_args()
    if args.phase == "response_transition":
        if args.parent_packet is None or args.parent_sha256 is None:
            parser.error("response_transition requires --parent-packet and --parent-sha256")
        result = run_response_transition(args.input, args.output, args.parent_packet, args.parent_sha256, args.device)
    else:
        result = run(args.input, args.output, args.device)
    print(json.dumps({"status": result["status"], "phase": result["phase"],
                      "error": result.get("error"), "jvp_validation": result.get("jvp_validation")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

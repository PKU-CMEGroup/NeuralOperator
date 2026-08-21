from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import train_pcno_euler2d_residual as base_trainer
from scripts.time_dependent_no.analyze_pcno_bump_gradient_ablation import (
    ENDPOINT_CALLS,
    FIXED_VISUAL_CASES,
    _seed_decision,
    render_fixed_case_residual_figures,
    render_summary_figures,
    run_analysis,
    verify_decomposition_directory,
)
from scripts.time_dependent_no.decompose_pcno_b1_frozen_rollout_error import (
    ADAPTER_PATH,
    DECOMPOSER_PATH,
    SUMMARY_KEY,
    compatibility_contract,
    compatibility_modules,
)
from scripts.time_dependent_no.preflight_pcno_bump_gradient_ablation import (
    A2_SCHEMA,
    a2_gate,
    validate_reference_contract,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_FROZEN_SOURCE_SHA256,
    B1_PARITY_KEYS,
    B1_VALIDATION_KEYS,
    DIFFERENTIAL_BRANCH_MODES,
    EXTENSION_PATHS,
    OPTIMIZER_STEPS_PER_PASS,
    PRESENTATIONS_PER_PASS,
    REGISTERED_COMPLETED_PASSES,
    REGISTERED_OPTIMIZER_STEPS,
    REGISTERED_PRESENTATIONS,
    REGISTERED_SCHEDULER_PASSES,
    REGISTERED_SEEDS,
    SCHEDULER_OPTIMIZER_STEPS,
    SCHEDULER_WARMUP_STEPS,
    _annotate_checkpoint,
    apply_differential_branch_mode,
    checkpoint_differential_branch_mode,
    extension_source_hashes,
    model_state_sha256,
    presentation_stream_sha256,
    registered_matched_config,
    registered_schedule_factor,
    validate_registered_args,
    verify_frozen_b1_sources,
    verify_training_data_manifest,
)
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)


def _unit_normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.zeros(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        mach_mean=0.0,
        mach_scale=1.0,
    )


def _synthetic_extension_hashes(tmp_path: Path) -> dict[str, str]:
    for relative in EXTENSION_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return extension_source_hashes(tmp_path)


def _small_model(seed: int = 1701) -> PCNOEuler2DResidual:
    torch.manual_seed(seed)
    return PCNOEuler2DResidual(
        normalization=_unit_normalization(),
        k_max=1,
        domain_lengths=(1.0, 1.0),
        layers=(4, 4),
        fc_dim=4,
        zero_initialize=False,
    )


def _b1_args(tmp_path: Path, seed: int = 20260718):
    return base_trainer.parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            str(seed),
            "--split-seed",
            "20260718",
            "--split-mode",
            "stratified",
            "--val-count",
            "30",
            "--step-stride",
            "1",
            "--epochs",
            "34",
            "--presentation-mode",
            "full_coverage",
            "--val-presentations",
            "128",
            "--batch-size",
            "4",
            "--stats-time-stride",
            "1",
            "--mach-scale-floor",
            "0.1",
            "--k-max",
            "8",
            "--domain-lengths",
            "6",
            "2",
            "--layers",
            "128",
            "128",
            "128",
            "128",
            "128",
            "--fc-dim",
            "128",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.00001",
            "--scheduler",
            "warmup_cosine",
            "--warmup-fraction",
            "0.02",
            "--warmup-start-factor",
            "0.1",
            "--min-learning-rate",
            "0.00002",
            "--gradient-clip",
            "1.0",
            "--input-noise-std",
            "0",
            "--generated-state-exposure-weight",
            "0",
            "--rollout-every",
            "1",
            "--rollout-val-count",
            "30",
            "--rollout-start-frame",
            "0",
            "--rollout-steps",
            "79",
            "--rollout-checkpoints",
            "20",
            "40",
            "60",
            "79",
            "--parity-rollout-keys",
            *B1_PARITY_KEYS,
            "--parity-rollout-horizon",
            "20",
            "--parity-max-rollout-relative-l2",
            "0.024284941703081132",
            "--parity-max-one-step-relative-l2",
            "0.005402445773142972",
            "--checkpoint-every",
            "1",
            "--boundary-mode",
            "causal_nodal_physical",
            "--boundary-max-source-hops",
            "3",
            "--boundary-rho-inf",
            "1.4",
            "--boundary-p-inf",
            "1.0",
            "--max-wall-hours",
            "23",
            "--device",
            "cuda",
            "--amp",
            "bf16",
        ]
    )


def test_registered_budget_and_scheduler_are_the_b1_prefix(tmp_path: Path) -> None:
    assert DIFFERENTIAL_BRANCH_MODES == ("full", "no_gradient")
    assert len(REGISTERED_SEEDS) == 3
    assert REGISTERED_COMPLETED_PASSES == 34
    assert REGISTERED_SCHEDULER_PASSES == 40
    assert REGISTERED_PRESENTATIONS == 725_220
    assert REGISTERED_OPTIMIZER_STEPS == 183_600
    assert PRESENTATIONS_PER_PASS == 21_330
    assert OPTIMIZER_STEPS_PER_PASS == 5_400
    assert SCHEDULER_OPTIMIZER_STEPS == 216_000
    assert SCHEDULER_WARMUP_STEPS == 4_320
    for step in (0, 1, 4_319, 4_320, 183_599, 215_999):
        expected = base_trainer.warmup_cosine_factor(
            step,
            total_steps=SCHEDULER_OPTIMIZER_STEPS,
            warmup_steps=SCHEDULER_WARMUP_STEPS,
            start_factor=0.1,
            minimum_factor=0.02,
        )
        assert registered_schedule_factor(step) == pytest.approx(expected, abs=1e-15)
    assert registered_schedule_factor(REGISTERED_OPTIMIZER_STEPS - 1) > 0.02

    args = _b1_args(tmp_path)
    validate_registered_args(args)
    config = registered_matched_config(args)
    assert config["completed_passes"] == 34
    assert config["scheduler_definition_passes"] == 40

    args.resume_checkpoint = tmp_path / "partial.pt"
    with pytest.raises(ValueError, match="does not register resume provenance"):
        validate_registered_args(args)


def test_training_manifest_check_closes_a_non_context_manager_store() -> None:
    closed: list[bool] = []

    class Store:
        manifest_digest = (
            "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
        )

        def __init__(self, _: Path) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    trainer = ModuleType("trainer")
    trainer.PCNOEuler2DShardStore = Store
    assert (
        verify_training_data_manifest(trainer, Path("unused")) == Store.manifest_digest
    )
    assert closed == [True]


def test_functional_no_gradient_is_exact_state_dict_stable_and_pairable() -> None:
    full = _small_model()
    no_gradient = _small_model()
    keys_before = tuple(no_gradient.state_dict())

    full_contract = apply_differential_branch_mode(full, "full")
    no_gradient_contract = apply_differential_branch_mode(no_gradient, "no_gradient")

    assert tuple(no_gradient.state_dict()) == keys_before
    assert (
        full_contract["initial_full_state_sha256"]
        == no_gradient_contract["initial_full_state_sha256"]
    )
    assert (
        full_contract["initial_nondifferential_state_sha256"]
        == no_gradient_contract["initial_nondifferential_state_sha256"]
    )
    assert no_gradient_contract["trainable_differential_parameters"] == 0
    assert no_gradient_contract["stored_differential_parameters"] > 0
    assert all(
        not parameter.requires_grad
        for module in no_gradient.backbone.gws
        for parameter in module.parameters()
    )

    x = torch.randn(2, 4, 7)
    edges = torch.zeros(2, 1, 2, dtype=torch.long)
    edge_weights = torch.zeros(2, 1, 2)
    assert torch.count_nonzero(no_gradient.backbone.gws[0](x, edges, edge_weights)) == 0

    # A fresh ordinary model loaded from the saved state also has an exact-zero
    # native differential response because gw2 is serialized at zero.
    reloaded = _small_model(seed=999)
    reloaded.load_state_dict(no_gradient.state_dict(), strict=True)
    assert torch.count_nonzero(reloaded.backbone.gws[0](x, edges, edge_weights)) == 0
    assert model_state_sha256(no_gradient) != model_state_sha256(full)


def test_b1_parameter_count_and_checkpoint_contract() -> None:
    model = PCNOEuler2DResidual(
        normalization=_unit_normalization(),
        k_max=8,
        domain_lengths=(6.0, 2.0),
        layers=(128, 128, 128, 128, 128),
        fc_dim=128,
        zero_initialize=True,
    )
    contract = apply_differential_branch_mode(model, "no_gradient")
    assert contract["stored_total_parameters"] == 19_155_720
    assert contract["stored_differential_parameters"] == 131_076
    assert contract["trainable_total_parameters"] == 19_024_644

    payload = {
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
    }
    annotated = _annotate_checkpoint(
        payload,
        model,
        "no_gradient",
        ["a", "b"],
        {"seed": 1},
    )
    assert checkpoint_differential_branch_mode(annotated) == "no_gradient"
    annotated["model_config"]["differential_branch_mode"] = "full"
    with pytest.raises(ValueError, match="declarations disagree"):
        checkpoint_differential_branch_mode(annotated)


def test_presentation_stream_hash_is_order_sensitive() -> None:
    first = [("7", 0), ("7", 1), ("16", 0)]
    second = [("7", 1), ("7", 0), ("16", 0)]
    assert presentation_stream_sha256(first) == presentation_stream_sha256(list(first))
    assert presentation_stream_sha256(first) != presentation_stream_sha256(second)


def test_frozen_source_verifier_accepts_exact_bytes_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {}
    for index, relative in enumerate(B1_FROZEN_SOURCE_SHA256):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"frozen-{index}\n".encode()
        path.write_bytes(payload)
        expected[relative] = __import__("hashlib").sha256(payload).hexdigest()
    monkeypatch.setattr(
        "scripts.time_dependent_no.train_pcno_bump_gradient_ablation.B1_FROZEN_SOURCE_SHA256",
        expected,
    )
    assert verify_frozen_b1_sources(tmp_path) == expected
    first = tmp_path / next(iter(expected))
    first.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_frozen_b1_sources(tmp_path)


def _arm_payload(multiplier: float) -> dict:
    by_call = {call: multiplier * (1.0 + call / 100.0) for call in ENDPOINT_CALLS}
    front_iou = {call: 0.8 / multiplier for call in ENDPOINT_CALLS}
    decomposition = {
        region: {
            "total_norm": multiplier,
            "propagated_norm": 0.8 * multiplier,
            "fresh_defect_norm": 0.2 * multiplier,
            "cross_energy_fraction_of_total": 0.0,
        }
        for region in (
            "normal_nodes_full",
            "boundary_nodes_full",
            "front_support_full",
            "smooth_highpass",
        )
    }
    return {
        "completion_rate": 1.0,
        "mean_survival_fraction": 1.0,
        "one_step_entry_relative_l2": multiplier,
        "state_relative_l2_by_call": by_call,
        "normal_pressure_rmse_by_call": by_call,
        "boundary_pressure_rmse_by_call": by_call,
        "smooth_highpass_by_call": by_call,
        "front_iou_by_call": front_iou,
        "front_chamfer_by_call": by_call,
        "front_position_by_call": by_call,
        "shock_thickness_log_error_by_call": by_call,
        "shock_strength_log_error_by_call": by_call,
        "decomposition_h79": decomposition,
    }


def test_seed_gate_and_summary_renderer(tmp_path: Path) -> None:
    full = _arm_payload(1.0)
    no_gradient = _arm_payload(0.9)
    decision = _seed_decision(full, no_gradient)
    assert decision["primary_h79_state_improves_at_least_5pct"] is True
    assert decision["all_controls_pass"] is True
    assert decision["seed_pass"] is True

    payloads = [
        {"seed": seed, "full": full, "no_gradient": no_gradient}
        for seed in REGISTERED_SEEDS
    ]
    paths = render_summary_figures(payloads, tmp_path)
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

    full["front_iou_by_call"][79] = 0.0
    no_gradient["front_iou_by_call"][79] = 0.0
    lost_front = _seed_decision(full, no_gradient)
    assert lost_front["controls"]["no_harm_front_iou"] is False
    assert lost_front["seed_pass"] is False

    with pytest.raises(ValueError, match="at least one registered seed"):
        run_analysis(tmp_path, tmp_path / "empty", seeds=(), render=False)
    with pytest.raises(ValueError, match="must be unique"):
        run_analysis(
            tmp_path,
            tmp_path / "duplicate",
            seeds=(REGISTERED_SEEDS[0], REGISTERED_SEEDS[0]),
            render=False,
        )
    with pytest.raises(ValueError, match="must lie"):
        run_analysis(tmp_path, tmp_path / "unknown", seeds=(1701,), render=False)


def test_decomposition_verifier_accepts_exact_record_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_sha256 = "checkpoint"
    decomposition_sources = {ADAPTER_PATH: "adapter", DECOMPOSER_PATH: "decomposer"}
    extension_sources = _synthetic_extension_hashes(tmp_path / "extension_sources")
    monkeypatch.setattr(
        "scripts.time_dependent_no.analyze_pcno_bump_gradient_ablation."
        "extension_source_hashes",
        lambda: extension_sources,
    )
    frozen_compatibility = compatibility_contract(
        frozen_sources=B1_FROZEN_SOURCE_SHA256,
        extension_sources=extension_sources,
        decomposition_sources=decomposition_sources,
    )
    summary = {
        "contract_complete": True,
        "checkpoint": {"sha256": checkpoint_sha256},
        "contract_checks": {"identity_closure": True},
        "source_files": decomposition_sources,
        SUMMARY_KEY: frozen_compatibility,
        "trajectories": [
            {"trajectory": trajectory, "completed": True}
            for trajectory in B1_VALIDATION_KEYS
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert (
        verify_decomposition_directory(tmp_path, checkpoint_sha256=checkpoint_sha256)
        == summary
    )

    summary["trajectories"] = {"count": len(B1_VALIDATION_KEYS)}
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(TypeError, match="record list"):
        verify_decomposition_directory(tmp_path, checkpoint_sha256=checkpoint_sha256)

    summary[SUMMARY_KEY]["maintained_decomposition_math_unchanged"] = False
    summary["trajectories"] = [
        {"trajectory": trajectory, "completed": True}
        for trajectory in B1_VALIDATION_KEYS
    ]
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="compatibility checks failed"):
        verify_decomposition_directory(tmp_path, checkpoint_sha256=checkpoint_sha256)


def test_frozen_decomposition_shims_bind_only_runtime_surfaces() -> None:
    evaluator = ModuleType("evaluator")
    trainer = ModuleType("trainer")
    evaluator.CAUSAL_BOUNDARY_MODE = "causal_nodal_physical"
    evaluator.build_model = object()
    evaluator.load_checkpoint = object()
    evaluator.select_device = object()
    trainer.contract_forward_sample = object()

    modules = compatibility_modules(evaluator, trainer)
    rollout = modules["utility.time_dependent_no.pcno_rollout"]
    runtime = modules["utility.time_dependent_no.pcno_runtime"]
    assert rollout.CAUSAL_BOUNDARY_MODE == "causal_nodal_physical"
    assert rollout.build_bump_checkpoint_model is evaluator.build_model
    assert rollout.load_bump_checkpoint is evaluator.load_checkpoint
    assert rollout.contract_forward_sample is trainer.contract_forward_sample
    assert runtime.select_device is evaluator.select_device


def _conservative_state(pressure: np.ndarray) -> np.ndarray:
    rho = np.ones_like(pressure)
    rho_u = np.zeros_like(pressure)
    rho_v = np.zeros_like(pressure)
    energy = pressure / 0.4
    return np.stack((rho, rho_u, rho_v, energy), axis=-1).astype(np.float32)


def test_fixed_case_residual_renderer_uses_paired_truth_and_common_artifacts(
    tmp_path: Path,
) -> None:
    seed = REGISTERED_SEEDS[0]
    nodes = 12
    x = np.linspace(0.0, 1.0, nodes)
    positions = np.column_stack((x, 0.1 * np.sin(2.0 * np.pi * x))).astype(np.float32)
    initial = _conservative_state(np.ones(nodes))
    targets = np.stack(
        [
            _conservative_state(1.0 + 0.01 * (call + 1) * np.sin(2.0 * np.pi * x))
            for call in range(79)
        ]
    )
    provenance = {str(seed): {}}
    for mode, perturbation in (("full", 0.001), ("no_gradient", 0.0005)):
        checkpoint_sha256 = f"{mode}-checkpoint"
        provenance[str(seed)][mode] = {"checkpoint_sha256": checkpoint_sha256}
        predictions = targets.copy()
        predictions[..., 3] += perturbation * np.arange(1, 80)[:, None]
        trajectory_dir = (
            tmp_path
            / f"seed_{seed}"
            / mode
            / "rollout_open_validation"
            / "trajectories"
        )
        trajectory_dir.mkdir(parents=True)
        for trajectory in FIXED_VISUAL_CASES:
            np.savez_compressed(
                trajectory_dir / f"trajectory_{trajectory}.npz",
                trajectory_key=np.asarray(trajectory),
                initial_conservative=initial,
                reference_targets_conservative=targets,
                pcno_baseline_predictions_conservative=predictions,
                positions=positions,
                checkpoint_sha256=np.asarray(checkpoint_sha256),
                test_manifest_digest=np.asarray(
                    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
                ),
                training_manifest_digest=np.asarray(
                    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
                ),
                baseline_valid_length=np.asarray(79),
            )
    output_dir = tmp_path / "figures"
    paths = render_fixed_case_residual_figures(
        tmp_path,
        output_dir,
        seed=seed,
        provenance=provenance,
    )
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_extension_scope_is_complete_and_strict_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="A1 extension files are missing"):
        extension_source_hashes()

    hashes = _synthetic_extension_hashes(tmp_path)
    assert len(hashes) == 5
    assert ADAPTER_PATH in hashes
    json.dumps(hashes, sort_keys=True, allow_nan=False)


def test_a2_reference_contract_and_gate_are_fail_closed() -> None:
    source_snapshot = {
        "source_set_digest": (
            "61ea3d2c650a0d5d67b2020c4a97a08dc5ffdfb55498da0a3b26129437c5b730"
        ),
        "files": {
            path: {"sha256": digest} for path, digest in B1_FROZEN_SOURCE_SHA256.items()
        },
    }
    split = {
        "data_manifest_digest": (
            "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
        ),
        "train_keys": [str(index) for index in range(270)],
        "val_keys": list(B1_VALIDATION_KEYS),
        "test_keys": [],
    }
    boundary = {
        "mode": "causal_nodal_physical",
        "policy_set_digest": (
            "c7d9f92dadeaef46bcb0c62c0f26458f08b7bfc6d6ec4e124aa840e9a246c13c"
        ),
        "trajectory_count": 300,
    }
    summary = {"data_manifest_digest": split["data_manifest_digest"]}
    reference = validate_reference_contract(
        summary=summary,
        split=split,
        boundary=boundary,
        source_snapshot=source_snapshot,
        run_contract={"source_snapshot": source_snapshot},
    )
    assert all(reference["checks"].values())
    assert reference["val_keys"] == B1_VALIDATION_KEYS

    arm = {
        "train_metrics": {"parameters_finite": True},
        "gradient_summary": {
            "all_finite": True,
            "differential_tensor_count": 1,
        },
        "call1_decomposition": {
            "identity_pass": True,
            "call1_zero_propagation_pass": True,
        },
        "native_evaluator": {"completion_rate": 1.0},
    }
    result = {
        "schema": A2_SCHEMA,
        "pairing": {
            "common_initial_full_state": True,
            "common_initial_nondifferential_state": True,
            "common_microbatch": True,
            "common_production_stream": True,
            "common_boundary_policy": True,
        },
        "arms": {
            "full": arm,
            "no_gradient": {
                **arm,
                "gradient_summary": {
                    "all_finite": True,
                    "differential_tensor_count": 0,
                },
            },
        },
    }
    assert all(a2_gate(result).values())
    result["arms"]["no_gradient"]["native_evaluator"]["completion_rate"] = 0.0
    assert a2_gate(result)["no_gradient_native_call_completed"] is False

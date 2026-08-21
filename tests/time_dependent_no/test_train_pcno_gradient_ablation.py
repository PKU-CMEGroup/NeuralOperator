from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.time_dependent_no.analyze_pcno_gradient_ablation import (
    SCHEMA as ANALYSIS_SCHEMA,
    run_analysis,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import build_model
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    SCHEMA,
    SEEDS,
    VARIANTS,
    FunctionalNoGradient,
    ParameterMatchedLocalReplacement,
    apply_variant,
    parameter_summary,
    registered_config,
    run_experiment,
    shared_state_sha256,
)


def _initialized_model(variant: str, seed: int = 1701) -> torch.nn.Module:
    config = registered_config(variant, seed, smoke=True, device_type="cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return apply_variant(build_model(config, torch.device("cpu")), variant)


def test_registered_matrix_has_three_paired_arms_and_fixed_budget() -> None:
    assert VARIANTS == ("full", "no_gradient", "local_replacement")
    assert SEEDS == (1701, 1702, 1703)
    configs = [
        registered_config(variant, seed, smoke=False, device_type="cuda")
        for variant in VARIANTS
        for seed in SEEDS
    ]

    assert len(configs) == 9
    assert {config["steps"] for config in configs} == {20_000}
    assert {tuple(config["resolution"]) for config in configs} == {(64, 32)}
    assert {config["primary_metric"] for config in configs} == {
        "native_held_phase_0p875_discontinuous_mean_relative_increment_l2"
    }
    assert len({config["working_run_id"] for config in configs}) == 9


def test_no_gradient_and_parameter_matched_local_semantics() -> None:
    full = _initialized_model("full")
    no_gradient = _initialized_model("no_gradient")
    local = _initialized_model("local_replacement")
    full_parameters = parameter_summary(full)
    no_gradient_parameters = parameter_summary(no_gradient)
    local_parameters = parameter_summary(local)
    width = 4
    expected_branch_parameters = 2 * width * width + 1

    assert isinstance(no_gradient.gws[0], FunctionalNoGradient)
    assert isinstance(local.gws[0], ParameterMatchedLocalReplacement)
    assert full_parameters["gradient_or_replacement_trainable"] == expected_branch_parameters
    assert no_gradient_parameters["gradient_or_replacement_trainable"] == 0
    assert local_parameters["gradient_or_replacement_trainable"] == expected_branch_parameters
    assert local_parameters["trainable"] == full_parameters["trainable"]
    assert no_gradient_parameters["trainable"] == (
        full_parameters["trainable"] - expected_branch_parameters
    )
    assert shared_state_sha256(full) == shared_state_sha256(no_gradient)
    assert shared_state_sha256(full) == shared_state_sha256(local)


def test_functional_no_gradient_returns_exact_zero() -> None:
    module = _initialized_model("no_gradient").gws[0]
    x = torch.randn(2, 4, 5)
    edges = torch.zeros(2, 1, 2, dtype=torch.long)
    weights = torch.zeros(2, 1, 2)

    output = module(x, edges, weights)

    assert torch.count_nonzero(output) == 0
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_local_replacement_inherits_model_device_and_dtype() -> None:
    config = registered_config(
        "local_replacement", 1701, smoke=True, device_type="cpu"
    )
    base = build_model(config, torch.device("cpu")).to(dtype=torch.float64)

    model = apply_variant(base, "local_replacement")

    assert all(
        parameter.device.type == "cpu" and parameter.dtype == torch.float64
        for module in model.gws
        for parameter in module.parameters()
    )


def test_three_arm_smoke_preserves_pairing_and_writes_parseable_artifacts(
    tmp_path: Path,
) -> None:
    summaries = {}
    contracts = {}
    for variant in VARIANTS:
        output_dir = tmp_path / f"{variant}_s1701"
        summaries[variant] = run_experiment(
            output_dir,
            variant=variant,
            seed=1701,
            smoke=True,
            device_name="cpu",
        )
        contracts[variant] = json.loads(
            (output_dir / "run_contract.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert summaries[variant]["schema"] == SCHEMA
        assert summaries[variant]["science_result"] is False
        assert summaries[variant]["completed_steps"] == 2
        assert summaries[variant]["primary_value"] >= 0.0
        assert (output_dir / "checkpoint.pt").is_file()
        assert (output_dir / "predictions.npz").is_file()
        assert (output_dir / "nested_predictions.npz").is_file()
        assert manifest["status"] == "completed"
        assert manifest["output_count"] == len(manifest["output_hashes"])
        json.dumps(summaries[variant], allow_nan=False)

    shared = {value["shared_initialization_sha256"] for value in contracts.values()}
    assert len(shared) == 1
    full_count = summaries["full"]["parameter_summary"]["trainable"]
    assert summaries["local_replacement"]["parameter_summary"]["trainable"] == full_count
    assert summaries["no_gradient"]["parameter_summary"]["trainable"] < full_count

    no_gradient_checkpoint = torch.load(
        tmp_path / "no_gradient_s1701" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    initialized = _initialized_model("no_gradient").state_dict()
    for name, value in no_gradient_checkpoint["model_state"].items():
        if name.startswith("gws."):
            torch.testing.assert_close(value, initialized[name], rtol=0.0, atol=0.0)

    analysis_dir = tmp_path / "analysis"
    analysis = run_analysis(tmp_path, analysis_dir, smoke=True)
    visual_manifest = json.loads(
        (analysis_dir / "visual_manifest.json").read_text(encoding="utf-8")
    )
    analysis_manifest = json.loads(
        (analysis_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert analysis["schema"] == ANALYSIS_SCHEMA
    assert analysis["science_result"] is False
    assert analysis["run_count"] == 3
    assert analysis["seeds"] == [1701]
    assert analysis["pairing_closures"]["all_passed"] is True
    assert analysis["pairing_closures"]["verified_input_output_hashes"] is True
    assert len(visual_manifest["files"]) == 14
    assert set(visual_manifest["files"]) == set(visual_manifest["sha256"])
    assert analysis_manifest["output_count"] == len(
        analysis_manifest["output_hashes"]
    )
    json.dumps(analysis, allow_nan=False)

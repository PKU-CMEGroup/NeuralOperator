"""Evaluate the fixed 16,384-update EMA Refiner with the shared stochastic assay.

Run python -m scripts.time_dependent_no.evaluate_kolmogorov_refiner
--training-input TRAIN --evaluation-input OPEN --fit-output FIT --output NEW
--phase assay|rollout --device cuda. No fitting or additional solver labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.time_dependent_no import evaluate_kolmogorov_acdm as common
from scripts.time_dependent_no import fit_kolmogorov_refiner as fit
from utility.time_dependent_no.pcno_kolmogorov_refiner import PeriodicVorticityRefiner, noise_tape

FIXED_UPDATES = 16384
DEPLOYMENT = "EMA; reverse levels 3,2,1,0; final physical-state restriction only"
SOURCE_PATHS = common.SOURCE_PATHS | fit.SOURCE_PATHS | {
    "scripts/time_dependent_no/evaluate_kolmogorov_refiner.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_refiner.py",
}


def load_model(directory, manifest, parent_fit, train_hash, device):
    directory = Path(directory)
    result = json.loads((directory/"result.json").read_text())
    identity = result["identity"]
    if (result["status"] != "completed" or result["updates_completed"] != FIXED_UPDATES
            or identity["updates"] != FIXED_UPDATES or identity["initial_update"] != 8192
            or identity["recipe"] != fit.RECIPE or identity["deployment"] != DEPLOYMENT
            or identity["input_manifest_sha256"] != train_hash or result["input_manifest"] != manifest
            or identity["model_config"] != parent_fit["model_config"]
            or identity["parent_checkpoint_sha256"] != manifest["artifacts"]["checkpoint.pt"]
            or set(result["sources"]) != fit.SOURCE_PATHS):
        raise ValueError("completed fixed Refiner identity mismatch")
    if any(common.base.sha256(common.base.ROOT/name) != digest for name, digest in result["sources"].items()):
        raise ValueError("acquisition source changed")
    if any(common.base.sha256(directory/name) != digest for name, digest in result["artifacts"].items()):
        raise ValueError("acquisition artifact changed")
    saved = torch.load(directory/"terminal.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != identity or saved["update"] != FIXED_UPDATES:
        raise ValueError("terminal identity mismatch")
    model = PeriodicVorticityRefiner(**identity["model_config"], train_scale=identity["train_scale"])
    model.load_state_dict(saved["ema"], strict=True)
    if float(model.train_scale) != identity["train_scale"]:
        raise ValueError("normalizer mismatch")
    return model.to(device).eval().requires_grad_(False), result


def run(training_input, evaluation_input, fit_output, output, phase, device="cuda"):
    return common.run(training_input, evaluation_input, fit_output, output, phase, device,
        model_loader=load_model, noise_factory=noise_tape, acquisition_probe=fit.acquisition_probe,
        source_paths=SOURCE_PATHS, calls_per_transition=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-input", "evaluation-input", "fit-output", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("assay", "rollout"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.training_input, args.evaluation_input, args.fit_output, args.output, args.phase, args.device)


if __name__ == "__main__":
    main()

from __future__ import annotations

MODEL_NAME = "geofno"

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baselines.geofno import GeoFNO, compute_Fourier_modes  # noqa: E402
from scripts.hifi3d.train_utils import (  # noqa: E402
    add_common_point_baseline_args,
    run_point_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GeoFNO on preprocessed HiFi3D point data.")
    add_common_point_baseline_args(parser)
    args = parser.parse_args()
    run_point_baseline(args, MODEL_NAME, _build_model)


def _build_model(args: argparse.Namespace, layers: list[int], Ls: list[float], in_dim: int, out_dim: int, device):
    modes = torch.tensor(
        compute_Fourier_modes(3, [args.k_max] * 3, Ls),
        dtype=torch.float32,
        device=device,
    )
    return GeoFNO(
        3,
        modes,
        layers=layers,
        fc_dim=args.fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        act="gelu",
    ).to(device)


if __name__ == "__main__":
    main()

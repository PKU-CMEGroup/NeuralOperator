from __future__ import annotations

MODEL_NAME = "pcno"

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.pcno import PCNO, PCNO_train, PCNO_train_multidist, compute_Fourier_modes  # noqa: E402
from scripts.hifi3d.train_utils import add_common_pcno_args, run_pcno_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCNO on preprocessed HiFi3D data.")
    add_common_pcno_args(parser)
    args = parser.parse_args()
    run_pcno_training(args, MODEL_NAME, _build_model, PCNO_train, PCNO_train_multidist)


def _build_model(args: argparse.Namespace, layers: list[int], Ls: list[float], in_dim: int, out_dim: int, device):
    modes = torch.tensor(
        compute_Fourier_modes(3, [args.k_max] * 3, Ls),
        dtype=torch.float32,
        device=device,
    )
    return PCNO(
        3,
        modes,
        nmeasures=1,
        layers=layers,
        fc_dim=args.fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        inv_L_scale_hyper=["together", 0.5, 2.0],
        act="gelu",
    ).to(device)


if __name__ == "__main__":
    main()

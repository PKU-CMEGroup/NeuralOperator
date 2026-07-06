from __future__ import annotations

MODEL_NAME = "mpcno"

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.mpcno import MPCNO, MPCNO_train, MPCNO_train_multidist, compute_Fourier_modes  # noqa: E402
from scripts.hifi3d.train_utils import (  # noqa: E402
    add_common_pcno_args,
    add_mpcno_args,
    run_pcno_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M-PCNO on preprocessed HiFi3D data.")
    add_common_pcno_args(parser)
    add_mpcno_args(parser)
    args = parser.parse_args()
    run_pcno_training(args, MODEL_NAME, _build_model, MPCNO_train, MPCNO_train_multidist)


def _build_model(args: argparse.Namespace, layers: list[int], Ls: list[float], in_dim: int, out_dim: int, device):
    modes = torch.tensor(
        compute_Fourier_modes(3, [args.k_max] * 3, Ls),
        dtype=torch.float32,
        device=device,
    )
    layer_selection = {
        "grad": args.grad.lower() == "true",
        "geo": args.geo.lower() == "true",
        "geointegral": args.geointegral.lower() == "true",
    }
    train_inv_L_scale = False if args.train_inv_L_scale == "False" else args.train_inv_L_scale
    return MPCNO(
        3,
        modes,
        nmeasures=1,
        layer_selection=layer_selection,
        layers=layers,
        fc_dim=args.fc_dim,
        in_dim=in_dim,
        out_dim=out_dim,
        inv_L_scale_hyper=[train_inv_L_scale, 0.5, 2.0],
        act="gelu",
        geo_act="softsign",
        scaling_mode="sqrt_inv",
    ).to(device)


if __name__ == "__main__":
    main()

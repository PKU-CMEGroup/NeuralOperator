from __future__ import annotations

MODEL_NAME = "transolver"

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baselines.transolver_plus import Model  # noqa: E402
from scripts.drivaerml.train_utils import (  # noqa: E402
    add_common_point_baseline_args,
    add_transolver_args,
    run_point_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transolver++ on preprocessed HiFi3D point data.")
    add_common_point_baseline_args(parser)
    add_transolver_args(parser)
    args = parser.parse_args()
    run_point_baseline(args, MODEL_NAME, _build_model)


def _build_model(args: argparse.Namespace, layers: list[int], Ls: list[float], in_dim: int, out_dim: int, device):
    return Model(
        space_dim=3,
        fun_dim=in_dim - 3,
        out_dim=out_dim,
        n_layers=len(layers),
        n_hidden=layers[0],
        n_head=args.transolver_nhead,
        dropout=args.transolver_dropout,
        mlp_ratio=args.transolver_mlp_ratio,
        slice_num=args.transolver_slice_num,
        ref=args.transolver_ref,
        unified_pos=False,
    ).to(device)


if __name__ == "__main__":
    main()

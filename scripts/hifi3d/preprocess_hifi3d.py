from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.geo_utility import compute_node_weights, preprocess_data_mesh
from scripts.hifi3d.hifi3d_helper import load_hifi3d_data, parse_dataset_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HiFi 3D VTP data for PCNO/M-PCNO.")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", type=str, default="BlendedNet,DrivAerML")
    parser.add_argument("--n_each", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("data/hifi3d_processed"))
    parser.add_argument("--output_name", type=str, default="smoke")
    parser.add_argument("--mesh_type", type=str, default="cell_centered", choices=["cell_centered", "vertex_centered"])
    parser.add_argument("--adjacent_type", type=str, default="edge", choices=["node", "edge", "face"])
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets={datasets}, n_each={args.n_each}, mesh_type={args.mesh_type}", flush=True)
    vertices_list, elems_list, features_list, names = load_hifi3d_data(
        args.data_root,
        datasets,
        args.n_each,
        args.seed,
        args.mesh_type,
    )
    print(f"Loaded {len(names)} samples", flush=True)

    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(
        vertices_list,
        elems_list,
        features_list,
        mesh_type=args.mesh_type,
        adjacent_type=args.adjacent_type,
    )
    node_measures, node_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure=False)

    npz_path = args.output_dir / f"{args.output_name}_{args.mesh_type}.npz"
    names_path = args.output_dir / f"{args.output_name}_names.npy"
    np.savez_compressed(
        npz_path,
        nnodes=nnodes,
        node_mask=node_mask,
        nodes=nodes,
        node_measures_raw=node_measures_raw,
        node_measures=node_measures,
        node_weights=node_weights,
        features=features,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
    )
    np.save(names_path, np.array(names, dtype=object))
    print(f"Saved {npz_path}", flush=True)
    print(f"Saved {names_path}", flush=True)


if __name__ == "__main__":
    main()

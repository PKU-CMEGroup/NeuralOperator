from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pcno.geo_utility import compute_node_weights, preprocess_data_mesh
from scripts.hifi3d.hifi3d_helper import load_hifi3d_data, parse_dataset_list


def list_vtp_files(data_root: Path, dataset: str) -> list[Path]:
    mesh_dir = data_root / f"{dataset}_20000"
    files = sorted(mesh_dir.glob("*.vtp"))
    if not files:
        raise FileNotFoundError(f"No .vtp files found in {mesh_dir}")
    return files


def geometry_hash(path: Path) -> str:
    mesh = pv.read(path)
    points = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    digest = hashlib.sha1()
    digest.update(points.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def write_blendednet_mod_manifest(files: list[Path], path: Path, mod_base: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file", "index", "index_mod", "index_minus1_mod"])
        for file in files:
            index = int(file.stem)
            writer.writerow([file.name, index, index % mod_base, (index - 1) % mod_base])


def select_subset_files(
    dataset: str,
    files: list[Path],
    large_n: int,
    blendednet_mode: str,
    blendednet_remainder: int,
    mod_base: int,
) -> tuple[list[Path], list[dict[str, object]]]:
    if dataset != "BlendedNet":
        if len(files) <= 1000:
            return files, []
        return files[: min(large_n, len(files))], []

    if blendednet_mode == "skip":
        return [], []

    if blendednet_mode == "fixed_mod_remainder":
        selected = [path for path in files if int(path.stem) % mod_base == blendednet_remainder]
        return selected[: min(large_n, len(selected))], []

    groups: dict[str, list[Path]] = defaultdict(list)
    for i, path in enumerate(files, start=1):
        groups[geometry_hash(path)].append(path)
        if i % 250 == 0:
            print(f"[BlendedNet] hashed {i}/{len(files)} files", flush=True)

    rows: list[dict[str, object]] = []
    for geom_hash, group_files in groups.items():
        group_files.sort()
        rows.append(
            {
                "geometry_hash": geom_hash,
                "group_size": len(group_files),
                "selected_file": group_files[0].name,
                "all_files": ",".join(path.name for path in group_files),
            }
        )
    rows.sort(key=lambda row: str(row["selected_file"]))

    selected = [Path(files[0]).with_name(str(row["selected_file"])) for row in rows]
    if len(selected) > large_n:
        selected = selected[:large_n]
    return selected, rows


def refresh_subset_dir(subset_root: Path, dataset: str, selected: list[Path]) -> Path:
    subset_dir = subset_root / f"{dataset}_20000"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for existing in subset_dir.glob("*.vtp"):
        existing.unlink()
    for path in selected:
        target = subset_dir / path.name
        target.symlink_to(path.resolve())
    return subset_dir


def write_selection_manifest(path: Path, selected: list[Path], group_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["selected_file", "source_path"])
        for selected_file in selected:
            writer.writerow([selected_file.name, str(selected_file)])

    if group_rows:
        group_path = path.with_name(path.stem + "_geometry_groups.tsv")
        with group_path.open("w", newline="") as f:
            fieldnames = ["geometry_hash", "group_size", "selected_file", "all_files"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(group_rows)


def prepare_subset_dirs(
    data_root: Path,
    subset_root: Path,
    datasets: list[str],
    large_n: int,
    blendednet_mode: str,
    blendednet_remainder: int,
    mod_base: int,
) -> list[str]:
    selected_datasets: list[str] = []
    summary = {}
    for dataset in datasets:
        files = list_vtp_files(data_root, dataset)
        if dataset == "BlendedNet":
            write_blendednet_mod_manifest(
                files,
                subset_root / "BlendedNet_mod_groups.tsv",
                mod_base,
            )
        selected, group_rows = select_subset_files(
            dataset,
            files,
            large_n,
            blendednet_mode,
            blendednet_remainder,
            mod_base,
        )
        if not selected:
            summary[dataset] = {
                "source_count": len(files),
                "selected_count": 0,
                "selection": "skipped",
                "mod_manifest": str(subset_root / "BlendedNet_mod_groups.tsv"),
            }
            print(f"{dataset}: source={len(files)} selected=0 skipped", flush=True)
            continue

        subset_dir = refresh_subset_dir(subset_root, dataset, selected)
        manifest = subset_root / f"{dataset}_selection.tsv"
        write_selection_manifest(manifest, selected, group_rows)
        duplicated_groups = sum(1 for row in group_rows if int(row["group_size"]) > 1)
        summary[dataset] = {
            "source_count": len(files),
            "selected_count": len(selected),
            "subset_dir": str(subset_dir),
            "manifest": str(manifest),
            "geometry_groups": len(group_rows),
            "duplicated_geometry_groups": duplicated_groups,
        }
        selected_datasets.append(dataset)
        print(f"{dataset}: source={len(files)} selected={len(selected)} subset={subset_dir}", flush=True)

    summary_path = subset_root / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}", flush=True)
    return selected_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare subsets and preprocess HiFi3D VTP data for baselines.")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", type=str, default="BlendedNet,DrivAerML")
    parser.add_argument("--n_each", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("data/hifi3d_processed"))
    parser.add_argument("--output_name", type=str, default="smoke")
    parser.add_argument("--mesh_type", type=str, default="cell_centered", choices=["cell_centered", "vertex_centered"])
    parser.add_argument("--adjacent_type", type=str, default="edge", choices=["node", "edge", "face"])
    parser.add_argument("--prepare_subsets", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--subset_root", type=Path, default=Path("data/HiFi3D_preprocess_ready"))
    parser.add_argument("--large_n", type=int, default=2000)
    parser.add_argument("--blendednet_mode", choices=["skip", "fixed_mod_remainder", "hash"], default="skip")
    parser.add_argument("--blendednet_remainder", type=int, default=1)
    parser.add_argument("--mod_base", type=int, default=8)
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    data_root = args.data_root
    if args.prepare_subsets or args.prepare_only:
        datasets = prepare_subset_dirs(
            args.data_root,
            args.subset_root,
            datasets,
            args.large_n,
            args.blendednet_mode,
            args.blendednet_remainder,
            args.mod_base,
        )
        data_root = args.subset_root
        if args.prepare_only:
            return
        if not datasets:
            raise ValueError("No datasets were selected for preprocessing.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets={datasets}, n_each={args.n_each}, mesh_type={args.mesh_type}", flush=True)
    vertices_list, elems_list, features_list, names = load_hifi3d_data(
        data_root,
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

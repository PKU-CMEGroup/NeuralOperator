from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

AUTO_MU_FIELDS = {
    "AirCraft": ["Ma", "alpha", "beta"],
    "NACA-CRM": ["Mach", "AlphaMean"],
    "BlendedNet": ["M_inf", "alpha_deg", "log10_Re", "alt_kft"],
}

LEAKY_MU_FIELDS = {
    "CA",
    "CN",
    "CZ",
    "Cl",
    "Cn",
    "Cm",
    "CD",
    "CL",
    "CMy",
    "c_d",
    "c_l",
    "c_my",
}


def _make_tensors(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    model_name: str,
    node_weights_array: np.ndarray,
    mu_array: np.ndarray | None,
):
    node_mask = torch.from_numpy(data["node_mask"][indices].astype(np.float32))
    nodes = torch.from_numpy(data["nodes"][indices].astype(np.float32))
    node_weights = torch.from_numpy(node_weights_array[indices].astype(np.float32))
    features = torch.from_numpy(data["features"][indices].astype(np.float32))
    directed_edges = torch.from_numpy(data["directed_edges"][indices].astype(np.int64))
    edge_gradient_weights = torch.from_numpy(data["edge_gradient_weights"][indices].astype(np.float32))

    normals = features[..., :3]
    x = torch.cat([normals, nodes], dim=-1)
    if mu_array is not None:
        mu = torch.from_numpy(mu_array[indices].astype(np.float32))
        mu_nodes = mu[:, None, :].expand(-1, nodes.shape[1], -1)
        x = torch.cat([x, mu_nodes], dim=-1)
        x = x * node_mask
    y = features[..., -1:]
    if model_name == "mpcno":
        aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, normals.permute(0, 2, 1))
    else:
        aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
    return x, y, aux


def _make_node_weights(data: np.lib.npyio.NpzFile, weight_mode: str, weight_factor: float) -> np.ndarray:
    if weight_mode == "normalized":
        return data["node_weights"]
    if weight_mode == "measure":
        node_measures = data["node_measures"]
        factor = weight_factor if weight_factor > 0 else float(np.max(np.sum(node_measures, axis=(1, 2))))
        if factor <= 0:
            raise ValueError("Cannot build measure weights from non-positive node measures")
        print(f"Using node_measures / factor with factor={factor}", flush=True)
        return node_measures / factor
    raise ValueError(f"Unsupported weight_mode: {weight_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCNO or M-PCNO on preprocessed HiFi 3D data.")
    parser.add_argument("--data_npz", type=Path, default=Path("data/hifi3d_processed/smoke_cell_centered.npz"))
    parser.add_argument("--names", type=Path, default=Path("data/hifi3d_processed/smoke_names.npy"))
    parser.add_argument("--model", type=str, default="mpcno", choices=["pcno", "mpcno"])
    parser.add_argument("--n_train", type=int, default=16)
    parser.add_argument("--n_test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--layer_sizes", type=str, default="16,16,16")
    parser.add_argument("--fc_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geo", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--weight_mode", type=str, default="normalized", choices=["normalized", "measure"])
    parser.add_argument("--weight_factor", type=float, default=0.0)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--train_inv_L_scale", type=str, default="False", choices=["False", "together", "independently"])
    parser.add_argument("--save_model_name", type=str, default="")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "metadata", "metadata_group", "metadata_condition"],
    )
    parser.add_argument("--split_group_field", type=str, default="geom_idx")
    parser.add_argument("--condition_fields", type=str, default="Ma,alpha,beta")
    parser.add_argument("--test_condition", type=str, default="")
    parser.add_argument("--use_mu", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--metadata_dir", type=Path, default=Path("data/HiFi3D_metadata"))
    parser.add_argument(
        "--mu_fields",
        type=str,
        default="auto",
        help=(
            "Comma-separated metadata fields to append as sample-level inputs. "
            "Use 'auto' for dataset defaults."
        ),
    )
    parser.add_argument("--mu_normalize", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--mu_missing", type=str, default="error", choices=["error", "zero"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_npz)
    names = np.load(args.names, allow_pickle=True) if args.names.exists() else None
    ndata = data["nodes"].shape[0]
    use_mu = args.use_mu.lower() == "true"
    metadata_split_modes = {"metadata", "metadata_group", "metadata_condition"}
    if args.split_mode == "random" and args.n_train + args.n_test > ndata:
        raise ValueError(f"Need n_train+n_test <= {ndata}, got {args.n_train + args.n_test}")
    if (use_mu or args.split_mode in metadata_split_modes) and names is None:
        raise ValueError("--use_mu True or metadata-based split modes require a valid --names file")

    metadata = None
    if use_mu or args.split_mode in metadata_split_modes:
        metadata = _load_metadata(args.metadata_dir)

    if args.split_mode == "metadata":
        train_idx, test_idx = _split_from_metadata(
            names,
            metadata,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
    elif args.split_mode == "metadata_group":
        train_idx, test_idx = _split_by_metadata_group(
            names,
            metadata,
            group_field=args.split_group_field,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
    elif args.split_mode == "metadata_condition":
        train_idx, test_idx = _split_by_metadata_condition(
            names,
            metadata,
            fields=[field.strip() for field in args.condition_fields.split(",") if field.strip()],
            test_condition=args.test_condition,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(ndata)
        train_idx = order[: args.n_train]
        test_idx = order[args.n_train : args.n_train + args.n_test]

    if names is not None:
        print("Train datasets:", _count_datasets(names[train_idx]), flush=True)
        print("Test datasets:", _count_datasets(names[test_idx]), flush=True)

    node_weights_array = _make_node_weights(data, args.weight_mode, args.weight_factor)
    mu_array = None
    if use_mu:
        mu_fields = _resolve_mu_fields(args.mu_fields, names)
        mu_array = _build_mu_array(names, metadata, mu_fields, missing=args.mu_missing)
        if args.mu_normalize.lower() == "true":
            mu_array, mu_mean, mu_std = _normalize_mu(mu_array, train_idx)
            print("Using normalized mu fields:", mu_fields, flush=True)
            print("mu mean:", np.array2string(mu_mean, precision=6), flush=True)
            print("mu std:", np.array2string(mu_std, precision=6), flush=True)
        else:
            print("Using raw mu fields:", mu_fields, flush=True)

    x_train, y_train, aux_train = _make_tensors(data, train_idx, args.model, node_weights_array, mu_array)
    x_test, y_test, aux_test = _make_tensors(data, test_idx, args.model, node_weights_array, mu_array)
    print(
        f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
        f"x_test={tuple(x_test.shape)} y_test={tuple(y_test.shape)}",
        flush=True,
    )

    layers = [int(size) for size in args.layer_sizes.split(",") if size]
    ndim = 3
    if args.Ls:
        Ls = [float(value) for value in args.Ls.split(",")]
        if len(Ls) != ndim:
            raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
    else:
        lengths = torch.amax(x_train[..., 3:6], dim=(0, 1)) - torch.amin(x_train[..., 3:6], dim=(0, 1))
        Ls = [max(2.0, float(length.item())) for length in lengths]
    print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}", flush=True)

    if args.model == "mpcno":
        from pcno.mpcno import MPCNO, MPCNO_train, compute_Fourier_modes

        modes = torch.tensor(compute_Fourier_modes(ndim, [args.k_max] * ndim, Ls), dtype=torch.float32).to(device)
        layer_selection = {
            "grad": args.grad.lower() == "true",
            "geo": args.geo.lower() == "true",
            "geointegral": args.geointegral.lower() == "true",
        }
        train_inv_L_scale = False if args.train_inv_L_scale == "False" else args.train_inv_L_scale
        model = MPCNO(
            ndim,
            modes,
            nmeasures=1,
            layer_selection=layer_selection,
            layers=layers,
            fc_dim=args.fc_dim,
            in_dim=x_train.shape[-1],
            out_dim=y_train.shape[-1],
            inv_L_scale_hyper=[train_inv_L_scale, 0.5, 2.0],
            act="gelu",
            geo_act="softsign",
            scaling_mode="sqrt_inv",
        ).to(device)
        train_fn = MPCNO_train
    else:
        from pcno.pcno import PCNO, PCNO_train, compute_Fourier_modes

        modes = torch.tensor(compute_Fourier_modes(ndim, [args.k_max] * ndim, Ls), dtype=torch.float32).to(device)
        model = PCNO(
            ndim,
            modes,
            nmeasures=1,
            layers=layers,
            fc_dim=args.fc_dim,
            in_dim=x_train.shape[-1],
            out_dim=y_train.shape[-1],
            inv_L_scale_hyper=["together", 0.5, 2.0],
            act="gelu",
        ).to(device)
        train_fn = PCNO_train

    config = {
        "train": {
            "base_lr": args.lr,
            "lr_ratio": 10,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "scheduler": "OneCycleLR",
            "batch_size": args.batch_size,
            "normalization_x": False,
            "normalization_y": True,
            "normalization_dim_x": [],
            "normalization_dim_y": [],
            "non_normalized_dim_x": 4,
            "non_normalized_dim_y": 0,
        }
    }

    save_model_name = args.save_model_name if args.save_model_name else None
    train_rel_l2, test_rel_l2, test_l2 = train_fn(
        x_train,
        aux_train,
        y_train,
        x_test,
        aux_test,
        y_test,
        config,
        model,
        save_model_name=save_model_name,
    )
    print("Final train_rel_l2:", train_rel_l2[-1], flush=True)
    print("Final test_rel_l2:", test_rel_l2[-1], flush=True)
    print("Final test_l2:", test_l2[-1], flush=True)


def _count_datasets(names: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        dataset = str(name).rsplit("-", 1)[0]
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def _split_from_metadata(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    *,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            raise KeyError(f"Missing metadata row for {name}")
        split = row.get("split", "")
        if split == "train":
            train_idx.append(i)
        elif split == "test":
            test_idx.append(i)
        else:
            raise ValueError(f"Metadata row for {name} has no train/test split")

    rng = np.random.default_rng(seed)
    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_array)
    rng.shuffle(test_idx_array)
    if n_train > len(train_idx_array):
        raise ValueError(f"Requested n_train={n_train}, but metadata split has {len(train_idx_array)} train samples")
    if n_test > len(test_idx_array):
        raise ValueError(f"Requested n_test={n_test}, but metadata split has {len(test_idx_array)} test samples")
    train_idx_array = train_idx_array[:n_train]
    test_idx_array = test_idx_array[:n_test]
    print(
        f"Using metadata split: train={len(train_idx_array)}/{len(train_idx)}, "
        f"test={len(test_idx_array)}/{len(test_idx)}",
        flush=True,
    )
    return train_idx_array, test_idx_array


def _split_by_metadata_group(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    *,
    group_field: str,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            raise KeyError(f"Missing metadata row for {name}")
        value = row.get(group_field, "")
        if value == "":
            raise KeyError(f"Missing metadata group field '{group_field}' for {name}")
        groups[value].append(i)

    rng = np.random.default_rng(seed)
    group_keys = np.asarray(sorted(groups, key=_sort_key), dtype=object)
    rng.shuffle(group_keys)

    test_idx: list[int] = []
    train_idx: list[int] = []
    test_groups: list[str] = []
    train_groups: list[str] = []
    for group_key in group_keys:
        group = groups[str(group_key)]
        if len(test_idx) + len(group) <= n_test:
            test_idx.extend(group)
            test_groups.append(str(group_key))
        else:
            train_idx.extend(group)
            train_groups.append(str(group_key))

    if len(test_idx) != n_test:
        raise ValueError(
            f"metadata_group split created {len(test_idx)} test samples, "
            f"but n_test={n_test}. Choose n_test compatible with group sizes."
        )
    if len(train_idx) < n_train:
        raise ValueError(f"metadata_group split has only {len(train_idx)} train samples, but n_train={n_train}")

    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_array)
    rng.shuffle(test_idx_array)
    train_idx_array = train_idx_array[:n_train]
    print(
        f"Using metadata group split on '{group_field}': "
        f"train_samples={len(train_idx_array)}, test_samples={len(test_idx_array)}, "
        f"train_groups={len(train_groups)}, test_groups={len(test_groups)}, "
        f"test_group_values={sorted(test_groups, key=_sort_key)}",
        flush=True,
    )
    return train_idx_array, test_idx_array


def _split_by_metadata_condition(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    *,
    fields: list[str],
    test_condition: str,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not fields:
        raise ValueError("--condition_fields must contain at least one field")
    condition = _parse_condition(fields, test_condition)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            raise KeyError(f"Missing metadata row for {name}")
        if all(_same_metadata_value(row.get(field, ""), value) for field, value in condition.items()):
            test_idx.append(i)
        else:
            train_idx.append(i)

    if len(train_idx) < n_train:
        raise ValueError(f"metadata_condition split has only {len(train_idx)} train samples, but n_train={n_train}")
    if len(test_idx) < n_test:
        raise ValueError(f"metadata_condition split has only {len(test_idx)} test samples, but n_test={n_test}")

    rng = np.random.default_rng(seed)
    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_array)
    rng.shuffle(test_idx_array)
    train_idx_array = train_idx_array[:n_train]
    test_idx_array = test_idx_array[:n_test]
    print(
        f"Using metadata condition split: train_samples={len(train_idx_array)}, "
        f"test_samples={len(test_idx_array)}, test_condition={condition}",
        flush=True,
    )
    return train_idx_array, test_idx_array


def _parse_condition(fields: list[str], test_condition: str) -> dict[str, str]:
    if not test_condition:
        raise ValueError("--test_condition is required for --split_mode metadata_condition")
    if "=" in test_condition:
        pairs = [item.strip() for item in test_condition.split(",") if item.strip()]
        condition: dict[str, str] = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if sep != "=" or not key.strip():
                raise ValueError(f"Invalid condition item: {pair}")
            condition[key.strip()] = value.strip()
    else:
        values = [value.strip() for value in test_condition.split(",") if value.strip()]
        if len(values) != len(fields):
            raise ValueError(
                f"Expected {len(fields)} values in --test_condition for fields {fields}, got {values}"
            )
        condition = dict(zip(fields, values))
    missing = [field for field in fields if field not in condition]
    if missing:
        raise ValueError(f"Condition is missing fields: {missing}")
    return {field: condition[field] for field in fields}


def _same_metadata_value(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    try:
        return abs(float(actual) - float(expected)) < 1.0e-9
    except ValueError:
        return False


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _load_metadata(metadata_dir: Path) -> dict[str, dict[str, str]]:
    paths = _metadata_paths(metadata_dir)
    records: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or "name" not in reader.fieldnames:
                raise ValueError(f"Metadata file must contain a 'name' column: {path}")
            for row in reader:
                name = row["name"]
                if name in records:
                    raise ValueError(f"Duplicate metadata row for {name}")
                records[name] = row
    print(f"Loaded metadata rows: {len(records)} from {len(paths)} file(s) under {metadata_dir}", flush=True)
    return records


def _metadata_paths(metadata_dir: Path) -> list[Path]:
    if metadata_dir.is_file():
        return [metadata_dir]
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata path not found: {metadata_dir}")

    patterns = [
        "*_metadata.tsv",
        "hifi3d_metadata.tsv",
        "*/data/hifi3d_metadata.tsv",
        "datasets/*/data/hifi3d_metadata.tsv",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(metadata_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No metadata TSV files found under {metadata_dir}")
    return paths


def _resolve_mu_fields(mu_fields: str, names: np.ndarray) -> list[str]:
    if mu_fields != "auto":
        fields = [field.strip() for field in mu_fields.split(",") if field.strip()]
    else:
        fields = []
        for dataset in _count_datasets(names):
            for field in AUTO_MU_FIELDS.get(dataset, []):
                if field not in fields:
                    fields.append(field)
        if not fields:
            raise ValueError("No auto mu fields are defined for the selected datasets")

    risky = sorted(set(fields).intersection(LEAKY_MU_FIELDS))
    if risky:
        print(f"Warning: selected mu fields look like output coefficients and may leak labels: {risky}", flush=True)
    return fields


def _build_mu_array(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    fields: list[str],
    *,
    missing: str,
) -> np.ndarray:
    mu = np.zeros((len(names), len(fields)), dtype=np.float32)
    missing_items: list[str] = []
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            if missing == "zero":
                missing_items.append(name)
                continue
            raise KeyError(f"Missing metadata row for {name}")
        for j, field in enumerate(fields):
            value = row.get(field, "")
            if value == "":
                if missing == "zero":
                    missing_items.append(f"{name}:{field}")
                    continue
                raise KeyError(f"Missing metadata field '{field}' for {name}")
            mu[i, j] = float(value)

    if missing_items:
        preview = ", ".join(missing_items[:5])
        print(f"Filled {len(missing_items)} missing mu values with zero. Examples: {preview}", flush=True)
    return mu


def _normalize_mu(mu: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_mean = mu[train_idx].mean(axis=0)
    mu_std = mu[train_idx].std(axis=0)
    mu_std = np.where(mu_std > 1.0e-6, mu_std, 1.0)
    return (mu - mu_mean) / mu_std, mu_mean, mu_std


if __name__ == "__main__":
    main()

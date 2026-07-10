from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utility.adam import Adam  # noqa: E402
from utility.losses import LpLoss  # noqa: E402
from utility.normalizer import UnitGaussianNormalizer  # noqa: E402


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


def _count_datasets(names: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        dataset = str(name).rsplit("-", 1)[0]
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def _group_indices_by_field(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]] | None,
    indices: np.ndarray,
    field: str,
) -> list[tuple[str, np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        name = str(names[index])
        value = ""
        if metadata is not None:
            row = metadata.get(name)
            if row is None:
                raise KeyError(f"Missing metadata row for {name}")
            value = row.get(field, "")
        if not value and field == "dataset":
            value = name.rsplit("-", 1)[0]
        if not value:
            raise KeyError(f"Missing metadata report field '{field}' for {name}")
        groups[value].append(int(index))

    result = [
        (label, np.asarray(group_indices, dtype=np.int64))
        for label, group_indices in groups.items()
    ]
    result.sort(key=lambda item: _sort_key(item[0]))
    print(
        f"Reporting test metrics by '{field}': "
        f"{ {label: len(group_indices) for label, group_indices in result} }",
        flush=True,
    )
    return result


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


def _split_balanced_by_metadata_field(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    *,
    field: str,
    n_train_per_group: int,
    n_test_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_train_per_group <= 0 or n_test_per_group <= 0:
        raise ValueError("--n_train_per_group and --n_test_per_group must both be positive")

    train_groups: dict[str, list[int]] = defaultdict(list)
    test_groups: dict[str, list[int]] = defaultdict(list)
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            raise KeyError(f"Missing metadata row for {name}")
        value = row.get(field, "")
        if not value and field == "dataset":
            value = name.rsplit("-", 1)[0]
        if not value:
            raise KeyError(f"Missing metadata balance field '{field}' for {name}")

        split = row.get("split", "")
        if split == "train":
            train_groups[value].append(i)
        elif split == "test":
            test_groups[value].append(i)
        else:
            raise ValueError(f"Metadata row for {name} has no train/test split")

    group_values = sorted(train_groups, key=_sort_key)
    missing_test_groups = sorted(set(group_values) - set(test_groups), key=_sort_key)
    if missing_test_groups:
        raise ValueError(f"Missing test groups for '{field}': {missing_test_groups}")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    group_counts: dict[str, dict[str, int]] = {}
    for group_value in group_values:
        train_candidates = np.asarray(train_groups[group_value], dtype=np.int64)
        test_candidates = np.asarray(test_groups[group_value], dtype=np.int64)
        if len(train_candidates) < n_train_per_group:
            raise ValueError(
                f"Group {group_value} has only {len(train_candidates)} train samples, "
                f"but n_train_per_group={n_train_per_group}"
            )
        if len(test_candidates) < n_test_per_group:
            raise ValueError(
                f"Group {group_value} has only {len(test_candidates)} test samples, "
                f"but n_test_per_group={n_test_per_group}"
            )

        rng.shuffle(train_candidates)
        rng.shuffle(test_candidates)
        selected_train = train_candidates[:n_train_per_group]
        selected_test = test_candidates[:n_test_per_group]
        train_idx.extend(selected_train.tolist())
        test_idx.extend(selected_test.tolist())
        group_counts[group_value] = {
            "train": int(len(selected_train)),
            "test": int(len(selected_test)),
        }

    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_array)
    rng.shuffle(test_idx_array)
    print(
        f"Using balanced metadata split on '{field}': "
        f"train_samples={len(train_idx_array)}, test_samples={len(test_idx_array)}, "
        f"group_counts={group_counts}",
        flush=True,
    )
    return train_idx_array, test_idx_array


def _split_ratio_by_metadata_field(
    names: np.ndarray,
    metadata: dict[str, dict[str, str]],
    *,
    field: str,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train_ratio_per_group must be in (0, 1)")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("--test_ratio_per_group must be in (0, 1)")
    if train_ratio + test_ratio >= 1.0:
        raise ValueError("--train_ratio_per_group + --test_ratio_per_group must be < 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, raw_name in enumerate(names):
        name = str(raw_name)
        row = metadata.get(name)
        if row is None:
            raise KeyError(f"Missing metadata row for {name}")
        value = row.get(field, "")
        if not value and field == "dataset":
            value = name.rsplit("-", 1)[0]
        if not value:
            raise KeyError(f"Missing metadata ratio field '{field}' for {name}")
        groups[value].append(i)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    group_counts: dict[str, dict[str, int]] = {}
    for group_value in sorted(groups, key=_sort_key):
        candidates = np.asarray(groups[group_value], dtype=np.int64)
        rng.shuffle(candidates)
        n_total = len(candidates)
        n_test = int(round(n_total * test_ratio))
        n_train = int(round(n_total * train_ratio))
        if n_test <= 0 or n_train <= 0:
            raise ValueError(
                f"Ratios produce an empty split for group {group_value}: "
                f"n_total={n_total}, n_train={n_train}, n_test={n_test}"
            )
        if n_train + n_test > n_total:
            raise ValueError(
                f"Ratios exceed group {group_value}: "
                f"n_total={n_total}, n_train={n_train}, n_test={n_test}"
            )

        selected_test = candidates[:n_test]
        selected_train = candidates[n_test : n_test + n_train]
        test_idx.extend(selected_test.tolist())
        train_idx.extend(selected_train.tolist())
        group_counts[group_value] = {
            "total": int(n_total),
            "train": int(n_train),
            "test": int(n_test),
        }

    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    test_idx_array = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx_array)
    rng.shuffle(test_idx_array)
    print(
        f"Using ratio metadata split on '{field}': "
        f"train_ratio={train_ratio}, test_ratio={test_ratio}, "
        f"train_samples={len(train_idx_array)}, test_samples={len(test_idx_array)}, "
        f"group_counts={group_counts}",
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


def add_common_pcno_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_npz", type=Path, default=Path("data/hifi3d_processed/smoke_cell_centered.npz"))
    parser.add_argument("--names", type=Path, default=Path("data/hifi3d_processed/smoke_names.npy"))
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
    parser.add_argument("--weight_mode", type=str, default="normalized", choices=["normalized", "measure"])
    parser.add_argument("--weight_factor", type=float, default=0.0)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--save_model_name", type=str, default="")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=[
            "random",
            "metadata",
            "metadata_group",
            "metadata_condition",
            "metadata_balanced",
            "metadata_ratio",
        ],
    )
    parser.add_argument("--split_group_field", type=str, default="geom_idx")
    parser.add_argument("--balance_field", type=str, default="dataset")
    parser.add_argument("--n_train_per_group", type=int, default=0)
    parser.add_argument("--n_test_per_group", type=int, default=0)
    parser.add_argument("--train_ratio_per_group", type=float, default=0.0)
    parser.add_argument("--test_ratio_per_group", type=float, default=0.0)
    parser.add_argument("--condition_fields", type=str, default="Ma,alpha,beta")
    parser.add_argument("--test_condition", type=str, default="")
    parser.add_argument(
        "--test_report_field",
        type=str,
        default="",
        help=(
            "Metadata field used to split test metrics into separate groups. "
            "Use 'dataset' for mixed HiFi3D runs."
        ),
    )
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


def add_mpcno_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grad", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geo", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--geointegral", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--train_inv_L_scale", type=str, default="False", choices=["False", "together", "independently"])


def run_pcno_training(args: argparse.Namespace, model_name: str, build_model, train_fn_single, train_fn_multidist) -> None:
    if model_name not in {"pcno", "mpcno"}:
        raise ValueError(f"Unsupported PCNO-family model: {model_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_npz)
    names = np.load(args.names, allow_pickle=True) if args.names.exists() else None
    ndata = data["nodes"].shape[0]
    use_mu = args.use_mu.lower() == "true"
    test_report_field = args.test_report_field.strip()
    metadata_split_modes = {
        "metadata",
        "metadata_group",
        "metadata_condition",
        "metadata_balanced",
        "metadata_ratio",
    }
    if args.split_mode == "random" and args.n_train + args.n_test > ndata:
        raise ValueError(f"Need n_train+n_test <= {ndata}, got {args.n_train + args.n_test}")
    if (use_mu or args.split_mode in metadata_split_modes) and names is None:
        raise ValueError("--use_mu True or metadata-based split modes require a valid --names file")
    if test_report_field and names is None:
        raise ValueError("--test_report_field requires a valid --names file")

    metadata = None
    if use_mu or args.split_mode in metadata_split_modes or (test_report_field and test_report_field != "dataset"):
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
    elif args.split_mode == "metadata_balanced":
        train_idx, test_idx = _split_balanced_by_metadata_field(
            names,
            metadata,
            field=args.balance_field,
            n_train_per_group=args.n_train_per_group,
            n_test_per_group=args.n_test_per_group,
            seed=args.seed,
        )
    elif args.split_mode == "metadata_ratio":
        train_idx, test_idx = _split_ratio_by_metadata_field(
            names,
            metadata,
            field=args.balance_field,
            train_ratio=args.train_ratio_per_group,
            test_ratio=args.test_ratio_per_group,
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

    x_train, y_train, aux_train = _make_tensors(data, train_idx, model_name, node_weights_array, mu_array)
    test_group_tensors = None
    if test_report_field:
        test_groups = _group_indices_by_field(names, metadata, test_idx, test_report_field)
        test_group_tensors = []
        for label, group_idx in test_groups:
            x_group, y_group, aux_group = _make_tensors(data, group_idx, model_name, node_weights_array, mu_array)
            test_group_tensors.append((label, x_group, y_group, aux_group))
        group_counts = {label: int(x_group.shape[0]) for label, x_group, _, _ in test_group_tensors}
        print(
            f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
            f"test_groups={group_counts}",
            flush=True,
        )
    else:
        x_test, y_test, aux_test = _make_tensors(data, test_idx, model_name, node_weights_array, mu_array)
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
        Ls = [float(length.item())*2+0.2 for length in lengths]
    print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using model={model_name}", flush=True)
    print(f"Using device={device}", flush=True)

    model = build_model(args, layers, Ls, x_train.shape[-1], y_train.shape[-1], device)
    train_fn = train_fn_multidist if test_group_tensors is not None else train_fn_single

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
    if test_group_tensors is not None:
        labels = [label for label, _, _, _ in test_group_tensors]
        x_test_list = [x_group for _, x_group, _, _ in test_group_tensors]
        y_test_list = [y_group for _, _, y_group, _ in test_group_tensors]
        aux_test_list = [aux_group for _, _, _, aux_group in test_group_tensors]
        train_rel_l2, test_rel_l2, test_l2 = train_fn(
            x_train,
            aux_train,
            y_train,
            x_test_list,
            aux_test_list,
            y_test_list,
            config,
            model,
            label_test_list=labels,
            save_model_name=save_model_name,
        )
    else:
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


def add_common_point_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_npz", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--n_train", type=int, default=16)
    parser.add_argument("--n_test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
    parser.add_argument("--fc_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--weight_mode", type=str, default="measure", choices=["normalized", "measure"])
    parser.add_argument("--weight_factor", type=float, default=0.0)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--save_model_name", type=str, default="")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=[
            "random",
            "metadata",
            "metadata_group",
            "metadata_condition",
            "metadata_balanced",
            "metadata_ratio",
        ],
    )
    parser.add_argument("--split_group_field", type=str, default="geom_idx")
    parser.add_argument("--balance_field", type=str, default="dataset")
    parser.add_argument("--n_train_per_group", type=int, default=0)
    parser.add_argument("--n_test_per_group", type=int, default=0)
    parser.add_argument("--train_ratio_per_group", type=float, default=0.0)
    parser.add_argument("--test_ratio_per_group", type=float, default=0.0)
    parser.add_argument("--condition_fields", type=str, default="Ma,alpha,beta")
    parser.add_argument("--test_condition", type=str, default="")
    parser.add_argument("--test_report_field", type=str, default="")
    parser.add_argument("--use_mu", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--metadata_dir", type=Path, default=Path("data/HiFi3D_metadata"))
    parser.add_argument("--mu_fields", type=str, default="auto")
    parser.add_argument("--mu_normalize", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--mu_missing", type=str, default="error", choices=["error", "zero"])
    parser.add_argument("--normalization_y", type=str, default="True", choices=["True", "False"])
    parser.add_argument(
        "--scheduler_step",
        type=str,
        default="auto",
        choices=["auto", "epoch", "batch"],
        help="OneCycleLR stepping. auto uses epoch for GeoFNO and batch for Transolver.",
    )
    parser.add_argument("--max_nodes", type=int, default=0, help="Optional node truncation for CPU smoke tests only.")


def add_transolver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transolver_nhead", type=int, default=8)
    parser.add_argument("--transolver_slice_num", type=int, default=32)
    parser.add_argument("--transolver_dropout", type=float, default=0.0)
    parser.add_argument("--transolver_mlp_ratio", type=int, default=2)
    parser.add_argument("--transolver_ref", type=int, default=8)
    parser.add_argument(
        "--transolver_condition_mode",
        type=str,
        default="repeat",
        choices=["repeat", "global"],
        help="repeat appends mu to every point; global passes mu through Transolver++ condition embedding.",
    )


def run_point_baseline(args: argparse.Namespace, model_name: str, build_model) -> None:
    if model_name not in {"geofno", "transolver"}:
        raise ValueError(f"Unsupported point baseline model: {model_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bundle = _load_point_baseline_tensors(args, model_name)
    x_train = bundle["x_train"]
    y_train = bundle["y_train"]
    aux_train = bundle["aux_train"]
    test_group_tensors = bundle["test_group_tensors"]
    x_test = bundle.get("x_test")
    y_test = bundle.get("y_test")
    aux_test = bundle.get("aux_test")

    layers = [int(size) for size in args.layer_sizes.split(",") if size]
    ndim = 3
    if args.Ls:
        Ls = [float(value) for value in args.Ls.split(",")]
        if len(Ls) != ndim:
            raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
    else:
        lengths = torch.amax(aux_train[1], dim=(0, 1)) - torch.amin(aux_train[1], dim=(0, 1))
        Ls = [max(2.0, float(length.item())) for length in lengths]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using model={model_name}", flush=True)
    print(f"Using Ls={Ls}, k_max={args.k_max}, layers={layers}", flush=True)
    print(f"Using device={device}", flush=True)

    model = build_model(args, layers, Ls, x_train.shape[-1], y_train.shape[-1], device)
    scheduler_step = args.scheduler_step
    if scheduler_step == "auto":
        scheduler_step = "batch" if model_name == "transolver" else "epoch"

    config = {
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "normalization_y": args.normalization_y.lower() == "true",
        "scheduler_step": scheduler_step,
    }

    save_model_name = args.save_model_name if args.save_model_name else None
    if test_group_tensors is None:
        train_rel_l2, test_rel_l2, test_l2 = train_baseline(
            model_name,
            x_train,
            aux_train,
            y_train,
            [(None, x_test, aux_test, y_test)],
            config,
            model,
            save_model_name=save_model_name,
        )
    else:
        labels_and_tensors = [
            (label, x_group, aux_group, y_group)
            for label, x_group, y_group, aux_group in test_group_tensors
        ]
        train_rel_l2, test_rel_l2, test_l2 = train_baseline(
            model_name,
            x_train,
            aux_train,
            y_train,
            labels_and_tensors,
            config,
            model,
            save_model_name=save_model_name,
        )

    print("Final train_rel_l2:", train_rel_l2[-1], flush=True)
    print("Final test_rel_l2:", test_rel_l2[-1], flush=True)
    print("Final test_l2:", test_l2[-1], flush=True)
    best_index = min(range(len(test_rel_l2)), key=lambda i: _score_test_loss(test_rel_l2[i]))
    print("Best epoch:", best_index, flush=True)
    print("Best test_rel_l2:", test_rel_l2[best_index], flush=True)


def _load_point_baseline_tensors(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    data = np.load(args.data_npz)
    names = np.load(args.names, allow_pickle=True) if args.names.exists() else None
    ndata = data["nodes"].shape[0]
    use_mu = args.use_mu.lower() == "true"
    test_report_field = args.test_report_field.strip()
    metadata_split_modes = {
        "metadata",
        "metadata_group",
        "metadata_condition",
        "metadata_balanced",
        "metadata_ratio",
    }
    if args.split_mode == "random" and args.n_train + args.n_test > ndata:
        raise ValueError(f"Need n_train+n_test <= {ndata}, got {args.n_train + args.n_test}")
    if (use_mu or args.split_mode in metadata_split_modes) and names is None:
        raise ValueError("--use_mu True or metadata-based split modes require a valid --names file")
    if test_report_field and names is None:
        raise ValueError("--test_report_field requires a valid --names file")

    metadata = None
    if use_mu or args.split_mode in metadata_split_modes or (test_report_field and test_report_field != "dataset"):
        metadata = _load_metadata(args.metadata_dir)

    if args.split_mode == "metadata":
        train_idx, test_idx = _split_from_metadata(names, metadata, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
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
    elif args.split_mode == "metadata_balanced":
        train_idx, test_idx = _split_balanced_by_metadata_field(
            names,
            metadata,
            field=args.balance_field,
            n_train_per_group=args.n_train_per_group,
            n_test_per_group=args.n_test_per_group,
            seed=args.seed,
        )
    elif args.split_mode == "metadata_ratio":
        train_idx, test_idx = _split_ratio_by_metadata_field(
            names,
            metadata,
            field=args.balance_field,
            train_ratio=args.train_ratio_per_group,
            test_ratio=args.test_ratio_per_group,
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

    transolver_condition_mode = getattr(args, "transolver_condition_mode", "repeat")
    x_train, y_train, aux_train = _make_baseline_tensors(
        data,
        train_idx,
        model_name,
        node_weights_array,
        mu_array,
        args.max_nodes,
        transolver_condition_mode,
    )
    result: dict[str, object] = {"x_train": x_train, "y_train": y_train, "aux_train": aux_train}

    if test_report_field:
        test_groups = _group_indices_by_field(names, metadata, test_idx, test_report_field)
        test_group_tensors = []
        for label, group_idx in test_groups:
            x_group, y_group, aux_group = _make_baseline_tensors(
                data,
                group_idx,
                model_name,
                node_weights_array,
                mu_array,
                args.max_nodes,
                transolver_condition_mode,
            )
            test_group_tensors.append((label, x_group, y_group, aux_group))
        group_counts = {label: int(x_group.shape[0]) for label, x_group, _, _ in test_group_tensors}
        print(f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} test_groups={group_counts}", flush=True)
        result["test_group_tensors"] = test_group_tensors
    else:
        x_test, y_test, aux_test = _make_baseline_tensors(
            data,
            test_idx,
            model_name,
            node_weights_array,
            mu_array,
            args.max_nodes,
            transolver_condition_mode,
        )
        print(
            f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)} "
            f"x_test={tuple(x_test.shape)} y_test={tuple(y_test.shape)}",
            flush=True,
        )
        result.update({"x_test": x_test, "y_test": y_test, "aux_test": aux_test, "test_group_tensors": None})

    return result


def _make_baseline_tensors(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    model_name: str,
    node_weights_array: np.ndarray,
    mu_array: np.ndarray | None,
    max_nodes: int,
    transolver_condition_mode: str,
):
    node_slice = slice(None if max_nodes <= 0 else max_nodes)
    node_mask = torch.from_numpy(data["node_mask"][indices, node_slice].astype(np.float32))
    nodes = torch.from_numpy(data["nodes"][indices, node_slice].astype(np.float32))
    node_weights = torch.from_numpy(node_weights_array[indices, node_slice].astype(np.float32))
    features = torch.from_numpy(data["features"][indices, node_slice].astype(np.float32))
    normals = features[..., :3]
    y = features[..., -1:] * node_mask

    condition = torch.empty((len(indices), 0), dtype=torch.float32)
    if model_name == "transolver":
        x = torch.cat([nodes, normals], dim=-1)
    else:
        x = torch.cat([normals, nodes], dim=-1)
    if mu_array is not None:
        mu = torch.from_numpy(mu_array[indices].astype(np.float32))
        if model_name == "transolver" and transolver_condition_mode == "global":
            if mu.shape[-1] != 3:
                raise ValueError("Transolver++ global condition expects exactly 3 mu fields.")
            condition = mu
        else:
            mu_nodes = mu[:, None, :].expand(-1, nodes.shape[1], -1)
            x = torch.cat([x, mu_nodes], dim=-1)
    x = x * node_mask
    aux = (node_mask, nodes, node_weights, condition)
    return x, y, aux


def train_baseline(
    model_name: str,
    x_train: torch.Tensor,
    aux_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    y_train: torch.Tensor,
    test_sets: list[tuple[str | None, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    config: dict[str, object],
    model: torch.nn.Module,
    save_model_name: str | None = None,
):
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["normalization_y"]:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim=0, normalization_dim=[])
        y_train = y_normalizer.encode(y_train)
        normalized_test_sets = []
        for label, x_test, aux_test, y_test in test_sets:
            normalized_test_sets.append((label, x_test, aux_test, y_normalizer.encode(y_test)))
        test_sets = normalized_test_sets
        y_normalizer.to(device)
    else:
        y_normalizer = None

    train_loader = _make_loader(x_train, y_train, aux_train, int(config["batch_size"]), shuffle=True)
    test_loaders = [
        (label, _make_loader(x_test, y_test, aux_test, int(config["batch_size"]), shuffle=False))
        for label, x_test, aux_test, y_test in test_sets
    ]

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=float(config["base_lr"]), weight_decay=float(config["weight_decay"]))
    scheduler_step = str(config["scheduler_step"])
    steps_per_epoch = len(train_loader) if scheduler_step == "batch" else 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["base_lr"]),
        div_factor=2,
        final_div_factor=100,
        pct_start=0.2,
        steps_per_epoch=steps_per_epoch,
        epochs=int(config["epochs"]),
    )
    loss_fn = LpLoss(d=1, p=2, size_average=False)

    for ep in range(int(config["epochs"])):
        t1 = default_timer()
        model.train()
        train_rel_l2 = 0.0
        n_train = len(train_loader.dataset)
        for x, y, node_mask, nodes, node_weights, condition in train_loader:
            x = x.to(device)
            y = y.to(device)
            node_mask = node_mask.to(device)
            nodes = nodes.to(device)
            node_weights = node_weights.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            out = _forward(model_name, model, x, node_mask, nodes, node_weights, condition)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask
            y = y * node_mask
            batch_size = x.shape[0]
            loss = loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler_step == "batch":
                scheduler.step()
            train_rel_l2 += loss.item()
        if scheduler_step == "epoch":
            scheduler.step()

        test_rel_l2, test_l2 = _evaluate(model_name, model, test_loaders, loss_fn, y_normalizer, device)
        train_rel_l2 /= n_train
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
        t2 = default_timer()
        print(
            "Epoch : ",
            ep,
            " Time: ",
            round(t2 - t1, 3),
            " Rel. Train L2 Loss : ",
            train_rel_l2,
            " Rel. Test L2 Loss : ",
            test_rel_l2,
            " Test L2 Loss : ",
            test_l2,
            flush=True,
        )
        if save_model_name and ((ep % 100 == 99) or (ep == int(config["epochs"]) - 1)):
            torch.save(model.state_dict(), save_model_name + ".pth")

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool,
):
    node_mask, nodes, node_weights, condition = aux
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y, node_mask, nodes, node_weights, condition),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _forward(model_name: str, model, x, node_mask, nodes, node_weights, condition):
    if model_name == "geofno":
        return model(x, (node_mask, nodes, node_weights))
    condition_arg = condition if condition.shape[-1] > 0 else None
    return model((x, nodes, condition_arg)) * node_mask


def _evaluate(model_name: str, model, test_loaders, loss_fn, y_normalizer, device):
    model.eval()
    results_rel: dict[str, float] = {}
    results_abs: dict[str, float] = {}
    with torch.no_grad():
        for label, loader in test_loaders:
            rel = 0.0
            abs_loss = 0.0
            n_test = len(loader.dataset)
            for x, y, node_mask, nodes, node_weights, condition in loader:
                x = x.to(device)
                y = y.to(device)
                node_mask = node_mask.to(device)
                nodes = nodes.to(device)
                node_weights = node_weights.to(device)
                condition = condition.to(device)
                out = _forward(model_name, model, x, node_mask, nodes, node_weights, condition)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out * node_mask
                y = y * node_mask
                batch_size = x.shape[0]
                rel += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                abs_loss += loss_fn.abs(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            key = label if label is not None else ""
            results_rel[key] = rel / n_test
            results_abs[key] = abs_loss / n_test

    if len(results_rel) == 1 and "" in results_rel:
        return results_rel[""], results_abs[""]
    return results_rel, results_abs


def _score_test_loss(value) -> float:
    if isinstance(value, dict):
        return float(np.mean(list(value.values())))
    return float(value)


if __name__ == "__main__":
    raise SystemExit("Use a concrete <model>_train.py entrypoint.")

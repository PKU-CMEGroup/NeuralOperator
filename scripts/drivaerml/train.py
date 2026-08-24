"""Shared memory-bounded data utilities for DrivAerML training scripts.

The preprocessed DrivAerML meshes have different numbers of cells and can be
hundreds of megabytes per field.  This module keeps the complete arrays on
disk with NumPy memory maps and materializes only a synchronized point sample
for each case.  Model-specific scripts can share the same record discovery,
train/test split, statistics, Dataset, and DataLoader implementation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_Y_FIELDS = ("CpMeanTrim", "wallShearStressMeanTrim")


def comma_separated(value: str) -> tuple[str, ...]:
    """Parse a non-empty, duplicate-free comma-separated field list."""
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    if not fields:
        raise argparse.ArgumentTypeError("At least one y field is required")
    if len(set(fields)) != len(fields):
        raise argparse.ArgumentTypeError(f"Repeated y field in {value!r}")
    return fields


@dataclass
class CacheRecord:
    """Paths and small statistics for one preprocessed DrivAerML case."""

    case_id: str
    source: Path
    metadata_path: Path
    node_data_path: Path
    y_paths: tuple[Path, ...]
    y_fields: tuple[str, ...]
    y_widths: tuple[int, ...]
    y_mean: np.ndarray
    y_m2: np.ndarray
    n_points: int
    n_cells: int
    total_measure: float
    cell_area_min: float
    cell_area_max: float
    coordinate_min: np.ndarray
    coordinate_max: np.ndarray

    @property
    def out_dim(self) -> int:
        return sum(self.y_widths)


def _case_id_from_metadata(path: Path) -> str:
    prefix = "metadata_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"Unexpected metadata filename: {path.name}")
    return path.stem[len(prefix) :]


def _metadata_array(
    value: object,
    width: int,
    label: str,
    metadata_path: Path,
) -> np.ndarray:
    """读取并验证 metadata 中固定长度的有限浮点数组。"""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(
            f"{metadata_path}: {label} must contain {width} finite values; "
            f"got {value!r}"
        )
    return array


def _metadata_cache_path(
    preprocess_dir: Path,
    filename: object,
    label: str,
    metadata_path: Path,
) -> Path:
    """解析 metadata 中的缓存文件名，但不打开大型 NumPy 数组。"""
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(
            f"{metadata_path}: {label}.file must be a local filename; "
            f"got {filename!r}"
        )
    path = preprocess_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{metadata_path}: missing {label} array {path}")
    return path


def discover_records(
    data_dir: Path,
    preprocess_dir: Path,
    y_fields: Sequence[str] = DEFAULT_Y_FIELDS,
    *,
    max_files: int = 0,
) -> list[CacheRecord]:
    """只读取小型 JSON metadata，建立每个 case 的缓存记录。

    这里不会调用 ``np.load``，也不会扫描大型几何/y 数组。数组 shape、dtype、
    几何范围和 y 统计量均由 ``preprocess.py`` 预先写入 metadata。
    """
    if max_files < 0:
        raise ValueError("max_files must be non-negative")

    data_dir = Path(data_dir).expanduser().resolve()
    preprocess_dir = Path(preprocess_dir).expanduser().resolve()
    y_fields = tuple(y_fields)
    if not y_fields:
        raise ValueError("At least one y field is required")
    metadata_paths = sorted(preprocess_dir.glob("metadata_*.json"))
    if max_files > 0:
        metadata_paths = metadata_paths[:max_files]
    if not metadata_paths:
        raise FileNotFoundError(
            f"No metadata_*.json files found in {preprocess_dir}. "
            "Run scripts/drivaerml/preprocess.py first."
        )

    records: list[CacheRecord] = []
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read metadata {metadata_path}: {error}") from error

        case_id = _case_id_from_metadata(metadata_path)
        if str(metadata.get("case_id")) != case_id:
            raise ValueError(
                f"{metadata_path}: case_id={metadata.get('case_id')!r} "
                f"does not match filename case {case_id!r}"
            )
        n_points = int(metadata["n_points"])
        n_cells = int(metadata["n_cells"])
        if n_points <= 0 or n_cells <= 0:
            raise ValueError(
                f"{metadata_path}: invalid n_points={n_points}, n_cells={n_cells}"
            )

        node_info = metadata["node_data"]
        if not isinstance(node_info, dict):
            raise ValueError(f"{metadata_path}: node_data must be an object")
        if node_info.get("shape") != [n_cells, 7]:
            raise ValueError(
                f"{metadata_path}: node_data.shape must be [{n_cells}, 7]; "
                f"got {node_info.get('shape')!r}"
            )
        try:
            np.dtype(node_info["dtype"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{metadata_path}: invalid node_data dtype") from error
        node_data_path = _metadata_cache_path(
            preprocess_dir, node_info.get("file"), "node_data", metadata_path
        )

        geometry = metadata["geometry"]
        if not isinstance(geometry, dict):
            raise ValueError(f"{metadata_path}: geometry must be an object")
        total_measure = float(geometry["total_measure"])
        cell_area_min = float(geometry["cell_area_min"])
        cell_area_max = float(geometry["cell_area_max"])
        coordinate_min = _metadata_array(
            geometry.get("coordinate_min"), 3, "coordinate_min", metadata_path
        )
        coordinate_max = _metadata_array(
            geometry.get("coordinate_max"), 3, "coordinate_max", metadata_path
        )
        if (
            not np.isfinite([total_measure, cell_area_min, cell_area_max]).all()
            or total_measure <= 0
            or cell_area_min < 0
            or cell_area_max < cell_area_min
            or np.any(coordinate_max < coordinate_min)
        ):
            raise ValueError(f"{metadata_path}: invalid geometry statistics")

        y_paths: list[Path] = []
        y_widths: list[int] = []
        y_means: list[np.ndarray] = []
        y_m2s: list[np.ndarray] = []
        available_y = metadata.get("y_fields")
        if not isinstance(available_y, dict):
            raise ValueError(f"{metadata_path}: y_fields must be an object")
        for field in y_fields:
            if field not in available_y:
                raise ValueError(
                    f"{metadata_path}: y field {field!r} was not preprocessed; "
                    f"available fields: {list(available_y)}"
                )
            y_info = available_y[field]
            if not isinstance(y_info, dict):
                raise ValueError(
                    f"{metadata_path}: metadata for y field {field!r} "
                    "must be an object"
                )
            shape = y_info.get("shape")
            if not isinstance(shape, list) or not shape or shape[0] != n_cells:
                raise ValueError(
                    f"{metadata_path}: invalid shape for y field {field!r}: {shape!r}"
                )
            width = 1 if len(shape) == 1 else int(shape[1]) if len(shape) == 2 else 0
            if width <= 0 or int(y_info.get("width", 0)) != width:
                raise ValueError(
                    f"{metadata_path}: invalid width for y field {field!r}"
                )
            try:
                np.dtype(y_info["dtype"])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"{metadata_path}: invalid dtype for y field {field!r}"
                ) from error
            path = _metadata_cache_path(
                preprocess_dir, y_info.get("file"), f"y field {field!r}", metadata_path
            )
            y_paths.append(path)
            y_widths.append(width)
            y_mean = _metadata_array(
                y_info.get("mean"), width, f"{field}.mean", metadata_path
            )
            y_m2 = _metadata_array(
                y_info.get("m2"), width, f"{field}.m2", metadata_path
            )
            if np.any(y_m2 < 0):
                raise ValueError(f"{metadata_path}: {field}.m2 must be non-negative")
            y_means.append(y_mean)
            y_m2s.append(y_m2)

        records.append(
            CacheRecord(
                case_id=case_id,
                source=data_dir / metadata.get("source_file", f"boundary_{case_id}.vtp"),
                metadata_path=metadata_path,
                node_data_path=node_data_path,
                y_paths=tuple(y_paths),
                y_fields=y_fields,
                y_widths=tuple(y_widths),
                y_mean=np.concatenate(y_means),
                y_m2=np.concatenate(y_m2s),
                n_points=n_points,
                n_cells=n_cells,
                total_measure=total_measure,
                cell_area_min=cell_area_min,
                cell_area_max=cell_area_max,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
            )
        )

        description = ", ".join(
            f"{field}:{width}"
            for field, width in zip(y_fields, y_widths)
        )
        print(
            f"case={case_id} cells={n_cells:,} "
            f"area={total_measure:.8g} "
            f"cell_area=[{cell_area_min:.3e}, {cell_area_max:.3e}] "
            f"y=[{description}]",
            flush=True,
        )

    expected_widths = records[0].y_widths
    if any(record.y_widths != expected_widths for record in records[1:]):
        raise ValueError("Y channel counts differ between cases")
    return records


def split_records(
    records: list[CacheRecord], n_train: int, n_test: int
) -> tuple[list[CacheRecord], list[CacheRecord]]:
    """Use the first cases for training and the last cases for testing."""
    ndata = len(records)
    if n_train <= 0 or n_test <= 0 or n_train + n_test > ndata:
        raise ValueError(
            f"Invalid split n_train={n_train}, n_test={n_test} for {ndata} cases"
        )
    return records[:n_train], records[-n_test:]


def y_statistics(records: Sequence[CacheRecord]) -> tuple[np.ndarray, np.ndarray]:
    """从各 case 的 metadata 合并 y 统计量，不打开大型 y 数组。"""
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    for record in records:
        case_count = record.n_cells
        case_mean = record.y_mean
        case_m2 = record.y_m2
        if mean is None:
            mean = case_mean.copy()
            m2 = case_m2.copy()
            count = case_count
            continue
        assert m2 is not None
        delta = case_mean - mean
        combined_count = count + case_count
        mean += delta * (case_count / combined_count)
        m2 += case_m2 + np.square(delta) * count * case_count / combined_count
        count = combined_count
    if count == 0 or mean is None or m2 is None:
        raise ValueError("Cannot compute y statistics from an empty split")
    return mean, np.maximum(np.sqrt(m2 / count), 1.0e-8)


def y_channel_names(record: CacheRecord) -> list[str]:
    """Expand vector y fields into stable, human-readable channel names."""
    names: list[str] = []
    axes = ("x", "y", "z")
    for field, width in zip(record.y_fields, record.y_widths):
        if width == 1:
            names.append(field)
        elif width == 3:
            names.extend(f"{field}_{axis}" for axis in axes)
        else:
            names.extend(f"{field}_{index}" for index in range(width))
    return names


class DrivAerMLDataset(Dataset):
    """从磁盘 mmap 数组中同步抽取几何信息和监督输出 ``y``。

    每个 Dataset index 对应一个完整的 DrivAerML case，而不是一个 cell。
    ``__getitem__`` 会从该 case 的 ``N`` 个 cell 中抽取至多
    ``sample_size`` 个 cell，并保证坐标、面积、法向和所有 y 物理场使用完全
    相同的 cell 索引。

    返回的模型输入通道顺序固定为：

    ``x = [x, y, z, normal_x, normal_y, normal_z]``。

    此外还会分别返回 ``nodes``、``measures`` 和 ``normals``，因为 Fourier
    basis 必须使用原始坐标，几何积分必须使用原始外法向和面积权重，不能只从
    拼接后的 ``x`` 中隐式推断。

    所有 cell 面积统一除以训练集共用的 ``global_weight_scale``，从而保留不同
    case 之间的总表面积差异，同时把 Fourier 积分控制在合适的数值尺度。

    ``sample_weight_correction`` 决定如何补偿只抽取部分 cell：

    - ``count``：乘以 ``N/m``，是均匀无放回抽样下的无偏估计；
    - ``measure``：乘以 ``total_measure/sampled_measure``，使每轮抽样的总面积
      权重严格等于完整网格的总面积权重，通常具有更稳定的数值尺度。

    预处理文件可以保持 ``float64``。这里只会把本次抽到的少量数据转换为
    ``dtype``，不会把几百 MB 的完整 mmap 数组整体复制进 RAM。
    """

    def __init__(
        self,
        records: Sequence[CacheRecord],
        sample_size: int,
        *,
        global_weight_scale: float,
        seed: int = 0,
        resample_each_epoch: bool = True,
        sample_weight_correction: str = "count",
        dtype: np.dtype = np.float32,
    ) -> None:
        if not records:
            raise ValueError("Dataset requires at least one record")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if global_weight_scale <= 0:
            raise ValueError("global_weight_scale must be positive")
        if sample_weight_correction not in {"count", "measure"}:
            raise ValueError(
                "sample_weight_correction must be count or measure"
            )
        # records 只保存路径和少量统计量，不包含完整网格数组。
        self.records = list(records)
        self.sample_size = int(sample_size)
        # seed、epoch 和 case index 共同决定抽样结果，使抽样可复现。
        self.seed = int(seed)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.global_weight_scale = float(global_weight_scale)
        self.sample_weight_correction = sample_weight_correction
        self.dtype = np.dtype(dtype)
        self.epoch = 0
        # 首次访问 case 时才打开 mmap；之后复用句柄，避免每个 batch 重复打开文件。
        self._mmap_cache: dict[
            str, tuple[np.ndarray, list[np.ndarray]]
        ] = {}

    def __getstate__(self):
        """DataLoader 创建 worker 时不跨进程复制已打开的 mmap 句柄。"""
        state = self.__dict__.copy()
        # 每个 worker 会在第一次读取时自行打开 mmap，避免共享文件句柄问题。
        state["_mmap_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch，使训练集在新 epoch 使用另一组确定性的抽样点。"""
        self.epoch = int(epoch)

    def sampled_indices(self, index: int) -> np.ndarray:
        """返回一个 case 本轮使用的 cell 索引，且不进行有放回抽样。"""
        n_cells = self.records[index].n_cells
        nsampled = min(self.sample_size, n_cells)
        # 网格点数不足 sample_size 时读取全部有效 cell，剩余位置稍后 padding。
        if nsampled == n_cells:
            return np.arange(n_cells, dtype=np.int64)
        # 测试集 resample_each_epoch=False，因此测试点集始终固定为 epoch=0。
        epoch = self.epoch if self.resample_each_epoch else 0
        rng = np.random.default_rng(self.seed + 1_000_003 * epoch + index)
        selected = rng.choice(n_cells, size=nsampled, replace=False)
        # 排序后读取 mmap 通常比随机顺序访问磁盘更友好。
        selected.sort()
        return selected

    def _arrays(
        self, record: CacheRecord
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """按需打开一个 case 的几何数组和所有 y 数组。"""
        arrays = self._mmap_cache.get(record.case_id)
        if arrays is None:
            arrays = (
                # mmap_mode='r' 只建立磁盘映射，不会立即把完整文件载入 RAM。
                np.load(record.node_data_path, mmap_mode="r"),
                [np.load(path, mmap_mode="r") for path in record.y_paths],
            )
            self._mmap_cache[record.case_id] = arrays
        return arrays

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        record = self.records[index]
        node_data, y_arrays = self._arrays(record)
        selected = self.sampled_indices(index)
        nsampled = selected.size
        # 高级索引只复制 selected 对应的行，不会把整个 mmap 文件载入 RAM。
        # 保留这个小数组的原始精度，以便稳定计算 sampled_measure；随后再转换
        # 为训练使用的 dtype。
        sampled_geometry = node_data[selected]
        geometry = np.asarray(sampled_geometry, dtype=self.dtype)

        # 不同 case 的有效 cell 数可能不同。固定第一维为 sample_size 后，
        # DataLoader 才能把多个 case 堆叠为同一个 batch。
        nodes = np.zeros((self.sample_size, 3), dtype=self.dtype)
        measures = np.zeros((self.sample_size, 1), dtype=self.dtype)
        normals = np.zeros((self.sample_size, 3), dtype=self.dtype)
        y = np.zeros(
            (self.sample_size, record.out_dim), dtype=self.dtype
        )
        mask = np.zeros((self.sample_size, 1), dtype=self.dtype)
        # node_data 的列顺序由 preprocess.py 固定为：
        # [center_x, center_y, center_z, area, normal_x, normal_y, normal_z]。
        nodes[:nsampled] = geometry[:, :3]
        measures[:nsampled, 0] = geometry[:, 3]
        normals[:nsampled] = geometry[:, 4:7]

        # 标量 y 的磁盘形状可能是 [N]，向量 y 的形状可能是 [N, C]；
        # 统一转换为 [nsampled, C] 后沿通道维拼接。
        y_values = []
        for array in y_arrays:
            values = np.asarray(array[selected], dtype=self.dtype)
            y_values.append(
                values[:, None] if values.ndim == 1 else values
            )
        y[:nsampled] = np.concatenate(y_values, axis=1)
        # mask=1 表示真实 cell，mask=0 表示为了 batch 对齐而添加的 padding。
        mask[:nsampled] = 1

        # 所有 case 始终使用同一个 G，保留 case 间的物理总面积差异。
        measures[:nsampled] /= self.global_weight_scale
        if nsampled < record.n_cells:
            if self.sample_weight_correction == "count":
                # 均匀抽样积分：sum_{i in sample} f_i w_i * N/m
                # 是完整离散积分 sum_{i=1}^N f_i w_i 的无偏估计。
                correction = record.n_cells / nsampled
            else:
                # 面积比修正让常数函数的积分以及抽样权重总和每轮都精确一致。
                sampled_measure = float(
                    sampled_geometry[:, 3].sum(dtype=np.float64)
                )
                if not np.isfinite(sampled_measure) or sampled_measure <= 0:
                    raise ValueError(
                        f"Invalid sampled cell area for case {record.case_id}: "
                        f"{sampled_measure}"
                    )
                correction = record.total_measure / sampled_measure
            measures[:nsampled] *= correction

        # 模型输入通道顺序必须与训练和全点评估保持一致：坐标在前，法向在后。
        x = np.concatenate((nodes, normals), axis=1)
        # torch.from_numpy 不再复制上述小数组，并保留 self.dtype 对应的精度。
        return tuple(
            torch.from_numpy(array)
            for array in (x, y, mask, nodes, measures, normals)
        )


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader with consistent worker and device settings.

    Workers are deliberately not persistent: before each epoch the training
    script calls ``dataset.set_epoch(epoch)``, and newly created workers must
    receive that updated value to produce a new deterministic point sample.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    device = torch.device(device)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator,
    )

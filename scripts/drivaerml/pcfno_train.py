"""Train PCFNO from the memory-mapped DrivAerML preprocessing output.

``preprocess.py`` writes one ``node_data_<case>.npy`` array and one NumPy
array per y field. This script never stacks the full meshes into a dense
tensor. It memory-maps every case and reads only the points sampled for the
current epoch, which is important for the multi-million-cell meshes.

The default y has four channels::

    CpMeanTrim (1) + wallShearStressMeanTrim (3)

Each y channel is standardized using the training split only. Training
uses the same ``Adam``, ``LpLoss``, and ``UnitGaussianNormalizer`` utilities
as PCNO/MPCNO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from timeit import default_timer
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcno.pcno import compute_Fourier_modes
from pcno_scale.pcfno import PCFNO
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from scripts.drivaerml.train import (
    CacheRecord,
    DEFAULT_Y_FIELDS,
    DrivAerMLDataset,
    comma_separated,
    discover_records,
    make_dataloader,
    split_records,
    y_channel_names,
    y_statistics,
)


def train(
    args: argparse.Namespace,
    train_records: list[CacheRecord],
    test_records: list[CacheRecord],
    model: PCFNO,
    y_normalizer: UnitGaussianNormalizer,
    global_weight_scale: float,
) -> None:
    # 预处理数组保持 float64；Dataset 只把本次抽到的数据转换为 float32。
    device = next(model.parameters()).device
    # y_normalizer 由调用方创建并传入，训练函数只负责把统计量移动到模型设备。
    y_normalizer.to(device)

    # 训练集每个 epoch 重新抽点；测试集始终使用同一组抽样点，保证指标可比较。
    train_dataset = DrivAerMLDataset(
        train_records,
        args.train_sample_size,
        global_weight_scale=global_weight_scale,
        seed=args.seed,
        resample_each_epoch=True,
        sample_weight_correction=args.sample_weight_correction,
        dtype=np.float32,
    )
    test_dataset = DrivAerMLDataset(
        test_records,
        args.test_sample_size,
        global_weight_scale=global_weight_scale,
        seed=args.seed + 10_000_019,
        resample_each_epoch=False,
        sample_weight_correction=args.sample_weight_correction,
        dtype=np.float32,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    y_names = y_channel_names(train_records[0])
    if checkpoint is not None:
        # 防止把输出通道顺序不同的 checkpoint 静默加载到当前任务。
        saved_names = checkpoint.get("y_channel_names")
        if saved_names is not None and list(saved_names) != y_names:
            raise ValueError(
                f"Checkpoint y channels {saved_names} do not match {y_names}"
            )
        if "y_mean" in checkpoint and "y_std" in checkpoint:
            # 恢复训练必须继续使用 checkpoint 保存时的 y 标准化统计量。
            # checkpoint 中的 y_std 已包含原 normalizer 的 eps，因此这里令 eps=0。
            y_normalizer.mean = torch.as_tensor(
                checkpoint["y_mean"],
                dtype=torch.float32,
                device=device,
            )
            y_normalizer.std = torch.as_tensor(
                checkpoint["y_std"],
                dtype=torch.float32,
                device=device,
            )
            y_normalizer.eps = 0.0

    lengths = [float(value) for value in args.Ls.split(",")]
    y_mean_numpy = y_normalizer.mean.detach().cpu().numpy()
    y_std_numpy = (y_normalizer.std + y_normalizer.eps).detach().cpu().numpy()

    print(
        f"device={device} dtype=float32 "
        f"global_weight_scale={global_weight_scale:.8g} "
        f"sample_weight_correction={args.sample_weight_correction} "
        f"out_dim={train_records[0].out_dim} k_max={args.k_max} "
        f"nmodes={model.modes.shape[0]} Ls={lengths} layers={model.layers} "
        f"geointegral={model.layer_selection['geointegral']}",
        flush=True,
    )

    for name, mean, std in zip(y_names, y_mean_numpy, y_std_numpy):
        print(f"y {name}: mean={mean:.8g} std={std:.8g}", flush=True)


    loss_function = LpLoss(d=1, p=2, size_average=False)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr,weight_decay=args.weight_decay,)
    total_steps = args.ep * len(train_loader)

    # 按用户设定始终使用 OneCycleLR；每完成一次 optimizer.step() 就调用一次。
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.base_lr,
        total_steps=total_steps,
        div_factor=2,
        final_div_factor=100,
        pct_start=0.2,
    )

    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_scheduler = checkpoint.get("scheduler_state_dict")
        if saved_scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, args.ep):
        epoch_start = default_timer()
        # epoch 参与 Dataset 的随机种子，使每个 epoch 抽到不同但可复现的 cell。
        train_dataset.set_epoch(epoch)
        model.train()
        train_relative_lp = 0.0
        for x, y, mask, nodes, weights, normals in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            nodes = nodes.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            normals = normals.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # x 的通道是 [坐标(3), 外法向(3)]；nodes、weights、normals 作为
            # Fourier 积分和 geointegral 的辅助几何量单独传入模型。
            normalized_prediction = model(x, (mask, nodes, weights, normals))
            # 网络学习标准化空间中的 y。计算物理误差前先 decode，再用 mask
            # 清除 padding 位置，防止 padding 参与相对 Lp loss。
            prediction = y_normalizer.decode(normalized_prediction) * mask
            masked_y = y * mask
            batch_size = x.shape[0]
            loss = loss_function(
                prediction.reshape(batch_size, -1),
                masked_y.reshape(batch_size, -1),
            )
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            # OneCycleLR 的 step 频率是每个 batch 一次，而不是每个 epoch 一次。
            scheduler.step()
            train_relative_lp += float(loss.item())

        model.eval()
        test_relative_lp = 0.0
        test_lp = 0.0
        with torch.no_grad():
            for x, y, mask, nodes, weights, normals in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                nodes = nodes.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                normals = normals.to(device, non_blocking=True)

                normalized_prediction = model(x, (mask, nodes, weights, normals))
                prediction = y_normalizer.decode(normalized_prediction) * mask
                masked_y = y * mask
                batch_size = x.shape[0]
                prediction = prediction.reshape(batch_size, -1)
                masked_y = masked_y.reshape(batch_size, -1)
                test_relative_lp += float(
                    loss_function(prediction, masked_y).item()
                )
                test_lp += float(
                    loss_function.abs(prediction, masked_y).item()
                )

        elapsed = default_timer() - epoch_start
        print(
            f"epoch={epoch:04d} time={elapsed:.3f}s "
            f"train_rel_lp={train_relative_lp / len(train_dataset):.8f} "
            f"test_rel_lp={test_relative_lp / len(test_dataset):.8f} "
            f"test_lp={test_lp / len(test_dataset):.8f}",
            flush=True,
        )

        if args.model_name and (
            epoch == args.ep - 1 or (epoch + 1) % args.save_every == 0
        ):
            model_path = Path(args.model_name)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(model_path) + ".pth")
            scheduler_state = scheduler.state_dict()
            # 当前 PyTorch 会把一个 Python 函数写入 anneal_func；删除它后，
            # checkpoint 才能使用 weights_only=True 安全加载。新建 scheduler
            # 已自带同一个退火函数，因此恢复时无需保存该对象。
            scheduler_state.pop("anneal_func", None)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler_state,
                    "epoch": epoch,
                    "y_fields": list(train_records[0].y_fields),
                    "y_channel_names": y_names,
                    "y_mean": y_mean_numpy.tolist(),
                    "y_std": y_std_numpy.tolist(),
                    "Ls": lengths,
                    "global_weight_scale": global_weight_scale,
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                str(model_path) + "_checkpoint.pth",
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCFNO from large memory-mapped DrivAerML arrays"
    )
    parser.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data" / "drivaerml")
    parser.add_argument("--preprocess_dir", type=Path, default=None, help="Defaults to <data_dir>/preprocess",)
    parser.add_argument(
        "--y_fields",
        type=comma_separated,
        default=DEFAULT_Y_FIELDS,
        help="Comma-separated NumPy y field names",
    )
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--train_sample_size", type=int, default=8192)
    parser.add_argument("--test_sample_size", type=int, default=8192)
    parser.add_argument(
        "--sample_weight_correction",
        choices=("count", "measure"),
        default="count",
        help=(
            "Correct sampled cell weights using N/m (count) or "
            "total_measure/sampled_measure (measure)"
        ),
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--geointegral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--Ls", type=str, default="")
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
    parser.add_argument("--fc_dim", type=int, default=128)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--ep", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=5.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--to_divide_factor", type=float, default=20.0)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.preprocess_dir is None:
        args.preprocess_dir = args.data_dir / "preprocess"
    return args


def main() -> None:
    args = parse_args()
    if args.max_files < 0:
        raise ValueError("--max_files must be non-negative")
    if min(
        args.train_sample_size,
        args.test_sample_size,
        args.save_every,
        args.ep,
        args.bsz,
    ) <= 0:
        raise ValueError("sample/batch/epoch/save values must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    preprocess_dir = args.preprocess_dir.expanduser().resolve()
    print(f"Reading metadata from {preprocess_dir}", flush=True)
    records = discover_records(
        data_dir,
        preprocess_dir,
        args.y_fields,
        max_files=args.max_files,
    )
    train_records, test_records = split_records(
        records, args.n_train, args.n_test
    )
    print(
        f"split train={[record.case_id for record in train_records]} "
        f"test={[record.case_id for record in test_records]}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.Ls:
        lengths = [float(value) for value in args.Ls.split(",")]
        if len(lengths) != 3:
            raise ValueError("--Ls must contain exactly three values")
    else:
        lower = np.min(
            np.stack([record.coordinate_min for record in records]), axis=0
        )
        upper = np.max(
            np.stack([record.coordinate_max for record in records]), axis=0
        )
        lengths = (2.0 * (upper - lower) + 0.2).tolist()
    args.Ls = ",".join(str(value) for value in lengths)

    print(f"Ls={args.Ls} ", flush=True,)
    
    layers = [int(value) for value in args.layer_sizes.split(",")]
    if len(layers) < 2 or any(width <= 0 for width in layers):
        raise ValueError("--layer_sizes needs at least two positive widths")
    modes = torch.as_tensor(
        compute_Fourier_modes(
            3, [args.k_max, args.k_max, args.k_max], lengths
        ),
        dtype=torch.float32,
        device=device,
    )
    model = PCFNO(
        ndims=3,
        modes=modes,
        nmeasures=1,
        layers=layers,
        fc_dim=args.fc_dim,
        in_dim=6,
        out_dim=train_records[0].out_dim,
        act=args.act,
        layer_selection={"geointegral": args.geointegral},
    ).to(device=device, dtype=torch.float32)

    y_mean, y_std = y_statistics(train_records)

    print(f"y_mean={y_mean} " f"y_std={y_std} ", flush=True,)
    
    y_normalizer = UnitGaussianNormalizer.from_statistics(
        torch.as_tensor(y_mean, dtype=torch.float32),
        torch.as_tensor(y_std, dtype=torch.float32),
    ).to(device)

    # 所有 case 共用同一个 G。这样既保留不同车辆的总表面积差异，也避免
    # Fourier 积分的数值尺度随物理面积直接增大。
    global_weight_scale = args.to_divide_factor * max(
        record.total_measure for record in train_records
    )

    train(
        args,
        train_records,
        test_records,
        model,
        y_normalizer,
        global_weight_scale,
    )


if __name__ == "__main__":
    main()


''' 
python pcfno_train.py \
    --data_dir "../../data/drivaerml" \
    --preprocess_dir "../../data/drivaerml/preprocess" \
    --y_fields CpMeanTrim,wallShearStressMeanTrim \
    --train_sample_size 16384 \
    --test_sample_size 16384 \
    --sample_weight_correction measure \
    --n_train 1 \
    --n_test  1 \
    --num_workers 4 \
    --geointegral \
    --k_max 16 \
    --layer_sizes 64,64,64,64 \
    --fc_dim 128 \
    --bsz 1 \
    --ep 200 \
    --base_lr 5e-4 \
    --weight_decay 1e-4 \
    --to_divide_factor 20 \
    --save_every 100 \
    --model_name "PCFNO" \
    2>&1 | tee pcfno_train.log
'''
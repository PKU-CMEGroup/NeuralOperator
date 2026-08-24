# 操作指南
首先因为是从别的文件夹中迁移过来，可能难免有文件夹错误等问题，或许需要额外的修补。

### Step 1

下载好1~500(实际共484个.vtp数据)，放在`data/DrivAerML/data`下。
data/hifi3d_processed/test
### Step 2

在`scripts/drivaerml`文件夹下运行(或者提交sbatch)
```sh
bash decimate.sh
```
于是应当产生了`data/HiFi3D/DrivAerML_20000`文件夹

### Step 3

在`scripts/drivaerml`文件夹下运行(或者提交sbatch)
```sh
bash preprocess.sh
```

于是应当产生了`data/hifi3d_processed/test`文件夹其中有一个.npz和一个.npy

### Step 4

运行`scripts/drivaerml`文件夹里面的训练脚本，可以在.sh脚本中修改参数。

Slurm 环境下可以依次提交预处理和 PCFNO 训练任务：

```sh
sbatch scripts/drivaerml/preprocess.sh
sbatch scripts/drivaerml/pcfno_train.sh
```

### PCFNO：训练超大网格

先运行 `preprocess.py`。它为每个 case 保存 `metadata_<case>.json`，其中包括
数组文件名/shape/dtype、cell 数量、总面积、坐标范围、cell 面积范围，以及
每个 y 通道的 mean 和 M2。`discover_records` 和 train split 的 y 统计量只读取
这些小型 JSON，不打开或扫描大型 NumPy 数组。Dataset 真正抽样时才 memory-map
`node_data_<case>.npy` 和逐物理场 NumPy 文件，不会把完整网格堆进内存。每个
epoch 从每个网格抽取 `train_sample_size` 个 cell，并按所选方式修正 Fourier
积分权重。

所有 cell 面积始终除以训练集共用的
`G = to_divide_factor * max(train_case_total_measure)`；不再提供其他
`weight_mode`。这会保留不同 case 的总表面积差异。

`--sample_weight_correction` 控制抽样权重修正方式：`count` 使用 `N/m`
（默认且无偏），`measure` 使用 `total_measure/sampled_measure`，使每次抽样的
总面积权重与完整网格严格一致。对于 cell 面积差异较大的 DrivAerML 网格，
可以使用 `--sample_weight_correction measure` 获得更稳定的权重尺度。

默认同时学习 4 个输出通道：

- `CpMeanTrim`：1 个压力系数通道；
- `wallShearStressMeanTrim`：3 个壁面剪切应力通道。

预处理阶段为每个 case 分块计算 y 统计量；训练时只组合 train split 对应的
metadata，因此每个 y channel 仍然只使用训练集统计量独立标准化。训练 loss
使用 `utility/losses.py` 中的相对 `LpLoss`。预处理数组保持 `float64`，抽样后
统一转成 `float32` 进行训练。

通用的 mmap Dataset、case discovery、train/test split、目标统计和
DataLoader 工厂位于 `scripts/drivaerml/train.py`。其他模型可以直接使用
`DrivAerMLDataset`，不需要依赖 PCFNO 的训练脚本。

全量训练：

```sh
python scripts/drivaerml/pcfno_train.py \
    --data_dir data/drivaerml \
    --preprocess_dir data/drivaerml/preprocess \
    --y_fields CpMeanTrim,wallShearStressMeanTrim \
    --train_sample_size 8192 \
    --test_sample_size 8192 \
    --sample_weight_correction measure \
    --n_train 400 \
    --n_test 80 \
    --geointegral \
    --k_max 4 \
    --bsz 1 \
    --ep 200 \
    --model_name model/pcfno_drivaerml
```

仓库中的两个超大案例可用下面的轻量命令检查整个流程：

```sh
python scripts/drivaerml/pcfno_train.py \
    --data_dir data/drivaerml \
    --train_sample_size 256 \
    --test_sample_size 256 \
    --n_train 1 \
    --n_test 1 \
    --k_max 2 \
    --layer_sizes 8,8 \
    --fc_dim 8 \
    --bsz 1 \
    --ep 2
```

### PCFNO 全点测试

`pcfno_eval_full.py` 会使用网格中的全部 cell，而不是把网格拆成互不相关的抽样子集。每个网络层分两遍计算：第一遍分块遍历所有点并累计全局 Fourier 系数，第二遍利用同一组全局系数重建所有点。默认把预处理数组和隐藏特征放在 CPU RAM，仅在 GPU 上按 `chunk_size` 生成 Fourier bases：

```sh
python scripts/drivaerml/pcfno_eval_full.py \
    --checkpoint model/pcfno_drivaerml_checkpoint.pth \
    --data_dir data/drivaerml \
    --chunk_size 16384 \
    --storage ram \
    --output_dir out/pcfno_full
```

对于约 880 万个 cell，四个预处理数组约占 270 MB；64 通道隐藏特征约占 2.25 GB。内存不足时可以切换到磁盘 mmap：

```sh
python scripts/drivaerml/pcfno_eval_full.py \
    --checkpoint model/pcfno_drivaerml_checkpoint.pth \
    --storage mmap \
    --work_dir /path/to/large/fast/storage
```

默认评估训练时的 test split，并为每个网格保存完整的 `<name>_prediction.npy` 和汇总 `metrics.csv`。增加 `--split all` 可评估全部网格，增加 `--write_vtp` 可将预测与绝对误差写回 VTP cell data。

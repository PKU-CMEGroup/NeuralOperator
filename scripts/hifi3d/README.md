# HiFi3D Experiment Scripts

This folder contains HiFi3D preprocessing, training, and profiling entrypoints
for NeuralOperator baselines. Run the commands below from the `NeuralOperator`
repository root.

## Files

- `hifi3d_helper.py`: VTP loading, target extraction, and cell/vertex sample construction.
- `preprocess_data.py`: optional subset preparation plus VTP-to-`.npz` preprocessing.
- `preprocess_data.sh`: Slurm array launcher for full per-dataset preprocessing caches.
- `train_utils.py`: shared split, metadata, mu-field, tensor, grouped-test, and training utilities.
- `pcno_train.py`: PCNO training entrypoint.
- `mpcno_train.py`: M-PCNO training entrypoint.
- `geofno_train.py`: GeoFNO training entrypoint.
- `transolver_train.py`: Transolver++ training entrypoint.
- `model_resource_profile.py`: CUDA memory and parameter profiling for the model families.

## Data Layout

Raw meshes are expected under one root with one `<dataset>_20000` folder per
dataset:

```text
data/HiFi3D/
  AirCraft_20000/
  BlendedNet_20000/
  DrivAerML_20000/
  DrivAerNet++_20000/
  DrivAerStar_20000/
  NACA-CRM_20000/
```

Generated caches, logs, and checkpoints should stay under `data/`, `logs/`,
and `checkpoints/`.

## 1. Preprocess Data

Small local smoke cache:

```bash
python scripts/hifi3d/preprocess_data.py \
  --data_root data/HiFi3D \
  --datasets AirCraft \
  --n_each 20 \
  --seed 0 \
  --output_dir data/hifi3d_processed \
  --output_name smoke \
  --mesh_type cell_centered \
  --adjacent_type edge
```

Full cache for one dataset:

```bash
python scripts/hifi3d/preprocess_data.py \
  --data_root data/HiFi3D \
  --datasets AirCraft \
  --n_each 0 \
  --seed 0 \
  --output_dir data/hifi3d_processed/cache \
  --output_name AirCraft_full \
  --mesh_type cell_centered \
  --adjacent_type edge
```

Full per-dataset caches on Slurm:

```bash
sbatch scripts/hifi3d/preprocess_data.sh
```

Process only one dataset through the same launcher:

```bash
DATASET=AirCraft sbatch scripts/hifi3d/preprocess_data.sh
```

Create symlink subsets before preprocessing:

```bash
python scripts/hifi3d/preprocess_data.py \
  --data_root data/HiFi3D \
  --datasets AirCraft,BlendedNet,DrivAerML,DrivAerNet++,DrivAerStar,NACA-CRM \
  --prepare_subsets \
  --prepare_only \
  --subset_root data/HiFi3D_preprocess_ready \
  --large_n 2000 \
  --blendednet_mode fixed_mod_remainder \
  --blendednet_remainder 1 \
  --mod_base 8
```

Preprocess the prepared subset root:

```bash
python scripts/hifi3d/preprocess_data.py \
  --data_root data/HiFi3D_preprocess_ready \
  --datasets AirCraft,BlendedNet,DrivAerML,DrivAerNet++,DrivAerStar,NACA-CRM \
  --n_each 0 \
  --seed 0 \
  --output_dir data/hifi3d_processed \
  --output_name hifi3d_subset \
  --mesh_type cell_centered \
  --adjacent_type edge
```

The resulting training files are:

```text
data/hifi3d_processed/<output_name>_<mesh_type>.npz
data/hifi3d_processed/<output_name>_names.npy
```

## 2. Train Models

Set paths once, then run any model entrypoint:

```bash
DATA_NPZ=data/hifi3d_processed/cache/AirCraft_full_cell_centered.npz
NAMES=data/hifi3d_processed/cache/AirCraft_full_names.npy
META_DIR=data/HiFi3D_metadata
SAVE_DIR=checkpoints/hifi3d
mkdir -p "${SAVE_DIR}" logs/hifi3d
```

PCNO:

```bash
python scripts/hifi3d/pcno_train.py \
  --data_npz "${DATA_NPZ}" \
  --names "${NAMES}" \
  --metadata_dir "${META_DIR}" \
  --split_mode random \
  --n_train 1600 \
  --n_test 400 \
  --epochs 200 \
  --batch_size 8 \
  --k_max 12 \
  --layer_sizes 64,64,64,64 \
  --fc_dim 128 \
  --use_mu False \
  --save_model_name "${SAVE_DIR}/pcno_aircraft"
```

M-PCNO:

```bash
python scripts/hifi3d/mpcno_train.py \
  --data_npz "${DATA_NPZ}" \
  --names "${NAMES}" \
  --metadata_dir "${META_DIR}" \
  --split_mode random \
  --n_train 1600 \
  --n_test 400 \
  --epochs 200 \
  --batch_size 8 \
  --k_max 12 \
  --layer_sizes 64,64,64,64 \
  --fc_dim 128 \
  --grad True \
  --geo True \
  --geointegral True \
  --train_inv_L_scale False \
  --use_mu False \
  --save_model_name "${SAVE_DIR}/mpcno_aircraft"
```

GeoFNO:

```bash
python scripts/hifi3d/geofno_train.py \
  --data_npz "${DATA_NPZ}" \
  --names "${NAMES}" \
  --metadata_dir "${META_DIR}" \
  --split_mode random \
  --n_train 1600 \
  --n_test 400 \
  --epochs 200 \
  --batch_size 8 \
  --k_max 12 \
  --layer_sizes 64,64,64,64 \
  --fc_dim 128 \
  --normalization_y True \
  --scheduler_step epoch \
  --save_model_name "${SAVE_DIR}/geofno_aircraft"
```

Transolver++:

```bash
python scripts/hifi3d/transolver_train.py \
  --data_npz "${DATA_NPZ}" \
  --names "${NAMES}" \
  --metadata_dir "${META_DIR}" \
  --split_mode random \
  --n_train 1600 \
  --n_test 400 \
  --epochs 200 \
  --batch_size 1 \
  --layer_sizes 384,384,384,384 \
  --transolver_nhead 8 \
  --transolver_slice_num 32 \
  --transolver_mlp_ratio 2 \
  --normalization_y True \
  --scheduler_step batch \
  --save_model_name "${SAVE_DIR}/transolver_aircraft"
```

For mixed-dataset runs, report each test distribution separately with
`--test_report_field dataset`. Metadata-based ratio splits need metadata rows
for all samples:

```bash
python scripts/hifi3d/mpcno_train.py \
  --data_npz data/hifi3d_processed/hifi3d_subset_cell_centered.npz \
  --names data/hifi3d_processed/hifi3d_subset_names.npy \
  --metadata_dir data/HiFi3D_metadata \
  --split_mode metadata_ratio \
  --balance_field dataset \
  --train_ratio_per_group 0.7 \
  --test_ratio_per_group 0.2 \
  --test_report_field dataset \
  --epochs 200 \
  --batch_size 8 \
  --k_max 12 \
  --layer_sizes 64,64,64,64 \
  --fc_dim 128 \
  --use_mu True \
  --save_model_name checkpoints/hifi3d/mpcno_mixed
```

Use `--split_mode metadata`, `metadata_group`, `metadata_condition`,
`metadata_balanced`, or `metadata_ratio` when the metadata files define the
desired split. Use `--use_mu True` only when the corresponding metadata fields
are available.

## 3. Profile Resources

`model_resource_profile.py` measures CUDA memory and parameter counts for the
main model families. Update the hard-coded data paths near the top of that file
or adapt them to your current cache before running:

```bash
python scripts/hifi3d/model_resource_profile.py
```

## Conventions

Keep shared training behavior in `train_utils.py`. Keep model-specific names,
argument surfaces, and model construction in the concrete `<model>_train.py`
entrypoint. Do not commit generated caches, checkpoints, Slurm logs, or local
experiment outputs.

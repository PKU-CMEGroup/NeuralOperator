#!/bin/bash
#SBATCH -o out/TRANSPP_DrivAerML.out
#SBATCH --qos=low
#SBATCH -J TRANSPP_DrivAerML
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

DATA_NPZ="../../data/hifi3d_processed/test/drivaerml_vertex_centered.npz"
SAVE_DIR="model/drivaerml_vertex_20k"

mkdir -p "$SAVE_DIR"
mkdir -p log

python transolver_train.py \
  --n_train 400 \
  --n_test 80 \
  --epochs 200 \
  --batch_size 1 \
  --layer_sizes 256,256,256,256 \
  --transolver_nhead 8 \
  --transolver_slice_num 32 \
  --transolver_mlp_ratio 2 \
  --normalization_y True \
  --scheduler_step batch \
  --save_model_name "${SAVE_DIR}/transolver" \
  > log/transolverpp.log

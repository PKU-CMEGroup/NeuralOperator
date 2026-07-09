#!/bin/bash
#SBATCH -o out/MPCNO.out
#SBATCH --qos=low
#SBATCH -J MPCNO
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

DATA_NPZ="../../data/hifi3d_processed/test/drivaerml_vertex_centered.npz"
NAMES="../../data/hifi3d_processed/test/drivaerml_names.npy"
META_DIR="../../data/hifi3d_processed/test/metadata/"
SAVE_DIR="model/drivaerml_vertex_20k/"

mkdir -p SAVE_DIR
mkdir -p log

python mpcno_train.py \
  --data_npz "${DATA_NPZ}" \
  --names "${NAMES}" \
  --metadata_dir "${META_DIR}" \
  --split_mode random \
  --n_train 400 \
  --n_test 80 \
  --epochs 200 \
  --batch_size 8 \
  --k_max 16 \
  --layer_sizes 64,64,64,64 \
  --fc_dim 128 \
  --Ls "4.1,2.1,1.5"\
  --grad True \
  --geo True \
  --geointegral True \
  --train_inv_L_scale False \
  --use_mu False \
  --save_model_name "${SAVE_DIR}/mpcno_drivaerml" \
  > log/mpcno_drivaerml_20k_k16_vertex_200_Ls412115.log
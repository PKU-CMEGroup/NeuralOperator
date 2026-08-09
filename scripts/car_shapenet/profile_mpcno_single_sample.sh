#!/bin/bash
#SBATCH -o out/MPCNO_profile_single_sample.out
#SBATCH --qos=low
#SBATCH -J MPCNO_profile
#SBATCH -p GPU80G
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate myconda

GRAD="True"
GEO="True"
GEOINTEGRAL="True"

LAYERS=(64 64 64 64 64 64 64)
ACT="gelu"
GEO_ACT="softsign"
K_MAX=16
LS="4.0,4.0,12.0"

MODEL_PATH="model/mpcno.pth"
DATA_PATH="../../data/car_shapenet"
DATA_INDEX=500
WARMUP=10
REPEATS=100
OUTPUT="eval_mpcno/mpcno_single_sample_timing.json"

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
mkdir -p log out eval_mpcno

python profile_mpcno_single_sample.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output "${OUTPUT}" \
    --data_index "${DATA_INDEX}" \
    --k_max "${K_MAX}" \
    --Ls "${LS}" \
    --layer_sizes "${LAYER_SIZES_STR}" \
    --act "${ACT}" \
    --geo_act "${GEO_ACT}" \
    --grad "${GRAD}" \
    --geo "${GEO}" \
    --geointegral "${GEOINTEGRAL}" \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    > "log/mpcno_profile_single_sample.log"

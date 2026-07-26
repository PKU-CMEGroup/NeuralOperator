#!/bin/bash
#SBATCH -o out/MPCNO_eval_errors.out
#SBATCH --qos=low
#SBATCH -J MPCNO_eval
#SBATCH -p GPU80G
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

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
OUTPUT_DIR="eval_mpcno"

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
mkdir -p log out model "${OUTPUT_DIR}"

python mpcno_eval_errors.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --k_max "${K_MAX}" \
    --Ls "${LS}" \
    --layer_sizes "${LAYER_SIZES_STR}" \
    --act "${ACT}" \
    --geo_act "${GEO_ACT}" \
    --grad "${GRAD}" \
    --geo "${GEO}" \
    --geointegral "${GEOINTEGRAL}" \
    > "log/mpcno_eval_errors.log"

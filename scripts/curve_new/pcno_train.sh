#!/bin/bash
#SBATCH -o out/PCNO_structured_%A.out
#SBATCH --qos=low
#SBATCH -J PCNO_structured
#SBATCH -p GPU80G
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3

# ========== params ==========
DATA_FILE="pcno_curve_data_1_1_5_2d_sp_laplace_panel.npz"
TWO_CIRCLES_DATA_FILE="pcno_curve_data_1_1_5_2d_sp_laplace_panel_two_circles.npz"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(64 64)
PROJ_LAYERS=(128 128 128)
ACT="none"

K_MAX_VALUES=(8 16 32 64)
K_MAX_COUNT=${#K_MAX_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}

EPOCHS=500
# =============================

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
PROJ_LAYER_SIZES_STR=$(IFS=,; echo "${PROJ_LAYERS[*]}")
LOG_DIR="log/pcno_structured/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch

python pcno_curve_test.py \
    --data_file $DATA_FILE \
    --two_circles_data_file $TWO_CIRCLES_DATA_FILE \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --k_max $K_MAX \
    --layer_sizes $LAYER_SIZES_STR \
    --proj_layer_sizes $PROJ_LAYER_SIZES_STR \
    --act $ACT \
    --ep $EPOCHS \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_proj${PROJ_LAYER_SIZES_STR}.log

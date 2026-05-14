#!/bin/bash
#SBATCH -o out/MPCNO_structured_%A.out
#SBATCH --qos=low
#SBATCH -J MPCNO_structured
#SBATCH -p GPU40G
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-3

# ========== params ==========
KERNEL_TYPE="dp_laplace"

GRAD="True"
GEO="True"
GEOINTEGRAL="True"

N_TRAIN=8000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(64 64)
PROJ_LAYERS=(128 128 128)
ACT="none"
GEO_ACT="softsign"
PROJ_ACT="gelu"

K_MAX_VALUES=(8 16 32 64)
K_MAX_COUNT=${#K_MAX_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID
K_MAX=${K_MAX_VALUES[$INDEX]}

EPOCHS=500
# =============================

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
PROJ_LAYER_SIZES_STR=$(IFS=,; echo "${PROJ_LAYERS[*]}")
LOG_DIR="log/mpcno_structured/${KERNEL_TYPE}/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch

python mpcno_curve_test.py \
    --grad $GRAD \
    --geo $GEO \
    --geointegral $GEOINTEGRAL \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --to_divide_factor $TO_DIVIDE_FACTOR \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --layer_sizes $LAYER_SIZES_STR \
    --proj_layer_sizes $PROJ_LAYER_SIZES_STR \
    --act $ACT \
    --geo_act $GEO_ACT \
    --proj_act $PROJ_ACT \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_proj${PROJ_LAYER_SIZES_STR}grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}.log
#!/bin/bash
#SBATCH -o out/PCNO_train_log_%A_%a.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-14  

# ========== params ==========

KERNEL_TYPES=("sp_laplace" "dp_laplace" "modified_dp_laplace" "adjoint_dp_laplace" "stokes")
N_VALUES=(1000 2000 4000)

GRAD="True"
GEO="True"
GEOINTEGRAL="True"

K_MAX=32  

N_TEST=1000
N_TWO_CIRCLES_TEST=1000

TO_DIVIDE_FACTOR=20.0
BATCH_SIZE=8
LAYERS=(64 64 64 64 64 64)
ACT="gelu"
GEO_ACT='softsign'

# =============================


KERNEL_COUNT=${#KERNEL_TYPES[@]}
N_COUNT=${#N_VALUES[@]}


INDEX=$SLURM_ARRAY_TASK_ID
KERNEL_INDEX=$((INDEX / N_COUNT))
N_INDEX=$((INDEX % N_COUNT))


KERNEL_TYPE=${KERNEL_TYPES[$KERNEL_INDEX]}
N_TRAIN=${N_VALUES[$N_INDEX]}

LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log/1_1_5_2d_${KERNEL_TYPE}/${LAYER_SIZES_STR}_${ACT}/"
mkdir -p ${LOG_DIR}

source activate pytorch 

python pcno_curve_geo_test.py \
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
    --act $ACT \
    --geo_act $GEO_ACT \
    --bsz $BATCH_SIZE \
    > ${LOG_DIR}/N${N_TRAIN}_Ntest${N_TEST},${N_TWO_CIRCLES_TEST}_k${K_MAX}_L10_bsz${BATCH_SIZE}_factor${TO_DIVIDE_FACTOR}_grad${GRAD}_geo${GEO}_geoint${GEOINTEGRAL}.log
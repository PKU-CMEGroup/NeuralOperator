#!/bin/bash
#SBATCH -o out/PCNO_train_log.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

# ========== 参数设置 ==========
KERNEL_TYPE="sp_laplace"
K_MAX=8
N_TRAIN=9000
N_TEST=1000
TWO_CIRCLES_TEST="True"
TWO_CIRCLES_TRAIN="False"
if [ "$KERNEL_TYPE" = "dp_laplace" ]; then
    NORMAL_PROD="True"
else
    NORMAL_PROD="False"
fi
TPYE="1_1_5_2d"
ACT="none"
LAYERS=(128 128)
NUM_LAYERS=${#LAYERS[@]}
LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
if [ "$TWO_CIRCLES_TEST" = "True" ]; then
    if [ "$TWO_CIRCLES_TRAIN" = "True" ]; then
        LOG_DIR="log/${TPYE}_${KERNEL_TYPE}_two_circles_train_test"
    else
        LOG_DIR="log/${TPYE}_${KERNEL_TYPE}_two_circles_test"
    fi
elif [ "$TWO_CIRCLES_TRAIN" = "True" ]; then
    LOG_DIR="log/${TPYE}_${KERNEL_TYPE}_two_circles_train"
else
    LOG_DIR="log/${TPYE}_${KERNEL_TYPE}"
fi

# 创建日志目录
mkdir -p ${LOG_DIR}

# =============================


source activate pytorch 

python pcno_curve_geo_test.py \
    --grad True \
    --geo False \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --two_circles_train $TWO_CIRCLES_TRAIN \
    --normal_prod $NORMAL_PROD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}.log

python pcno_curve_geo_test.py \
    --grad True \
    --geo True \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --two_circles_train $TWO_CIRCLES_TRAIN \
    --normal_prod $NORMAL_PROD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_geo3wx.log

python pcno_curve_geo_test.py \
    --grad False \
    --geo True \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --two_circles_train $TWO_CIRCLES_TRAIN \
    --normal_prod $NORMAL_PROD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_nograd_geo3wx.log

python pcno_curve_geo_test.py \
    --grad False \
    --geo False \
    --lap False \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --kernel_type $KERNEL_TYPE \
    --k_max $K_MAX \
    --two_circles_test $TWO_CIRCLES_TEST \
    --two_circles_train $TWO_CIRCLES_TRAIN \
    --normal_prod $NORMAL_PROD \
    --layer_sizes $LAYER_SIZES_STR \
    --act $ACT \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_nograd.log
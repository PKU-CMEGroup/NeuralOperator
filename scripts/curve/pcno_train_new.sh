#!/bin/bash
#SBATCH -o out/PCNO_train_log.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

# ========== 参数人工设置 ==========

# 数据参数
KERNEL_TYPE="sp_laplace" # 'sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace', 'fredholm_laplace', 'exterior_laplace_neumann'
K_MAX=8
N_TRAIN=9000
N_TEST=1000
N_TWO_CIRCLES_TEST=1000
NORMAL_PROD="False"
TPYE="1_1_5_2d"

# 模型参数
ACT="none"
GEO_ACT="gelu"
LAYERS=(128 128)
SCALE=0.0
ZERO_INIT="True"
IF_DEEP="True"
# geo_dims,num_grad,geo_grad 几个参数我这边还不太会设置,还得麻烦张 研疏学长

# 训练参数
EPOCHS=500
BATCH_SIZE=128
# 或许其他几个参数也需要根据这两个而改变吗?或许也需要设置?

# ========== 参数自动设置 ==========
NUM_LAYERS=${#LAYERS[@]}
LAYER_SIZES_STR=$(IFS=,; echo "${LAYERS[*]}")
LOG_DIR="log/${TPYE}_${KERNEL_TYPE}" # 还需要再添加别的参数进入这个名称吗?

# 创建日志目录
mkdir -p ${LOG_DIR}

# =============================
# 下面log的名称是不是还要再加上点参数?

source activate pytorch 

python pcno_curve_deep_geo_test.py \
    --grad True \
    --geo False \
    # geograd略过了先
    --lap False \
    # geodims略过了先
    # numgrad略过了先
    --k_max $K_MAX \
    --bsz $BATCH_SIZE \
    --ep $EPOCHS \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --act $ACT \
    --scale $SCALE \
    --geo_act $GEO_ACT \
    --zero_init $ZERO_INIT \
    --if_deep $IF_DEEP \
    --layer_sizes $LAYER_SIZES_STR \
    --normal_prod $NORMAL_PROD \
    --kernel_type $KERNEL_TYPE \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}.log

python pcno_curve_deep_geo_test.py \
    --grad True \
    --geo True \
    --lap False \
    --k_max $K_MAX \
    --bsz $BATCH_SIZE \
    --ep $EPOCHS \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --act $ACT \
    --scale $SCALE \
    --geo_act $GEO_ACT \
    --zero_init $ZERO_INIT \
    --if_deep $IF_DEEP \
    --layer_sizes $LAYER_SIZES_STR \
    --normal_prod $NORMAL_PROD \
    --kernel_type $KERNEL_TYPE \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_geo3wx.log

python pcno_curve_deep_geo_test.py \
    --grad False \
    --geo True \
    --lap False \
    --k_max $K_MAX \
    --bsz $BATCH_SIZE \
    --ep $EPOCHS \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --act $ACT \
    --scale $SCALE \
    --geo_act $GEO_ACT \
    --zero_init $ZERO_INIT \
    --if_deep $IF_DEEP \
    --layer_sizes $LAYER_SIZES_STR \
    --normal_prod $NORMAL_PROD \
    --kernel_type $KERNEL_TYPE \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_nograd_geo3wx.log

python pcno_curve_deep_geo_test.py \
    --grad False \
    --geo False \
    --lap False \
    --k_max $K_MAX \
    --bsz $BATCH_SIZE \
    --ep $EPOCHS \
    --n_train $N_TRAIN \
    --n_test $N_TEST \
    --n_two_circles_test $N_TWO_CIRCLES_TEST \
    --act $ACT \
    --scale $SCALE \
    --geo_act $GEO_ACT \
    --zero_init $ZERO_INIT \
    --if_deep $IF_DEEP \
    --layer_sizes $LAYER_SIZES_STR \
    --normal_prod $NORMAL_PROD \
    --kernel_type $KERNEL_TYPE \
    > ${LOG_DIR}/k${K_MAX}_L10_num_${NUM_LAYERS}_layer_${LAYER_SIZES_STR}_act_${ACT}_nograd.log
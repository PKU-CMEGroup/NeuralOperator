#!/bin/bash
#SBATCH -o out/MPCNO.out
#SBATCH --qos=low
#SBATCH -J MPCNO
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

mkdir -p log

python mpcno_train.py \
    --grad True \
    --geo  True \
    --geointegral True\
    --k_max 12 \
    --bsz 8 \
    --ep 200 \
    --n_train 400 \
    --n_test 80 \
    --layer_sizes 64,64,64,64 \
    --model_name "model/mpcno" \
    > log/mpcno.log
#!/bin/bash
#SBATCH -o out/GeoFNO.out
#SBATCH --qos=low
#SBATCH -J GeoFNO
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

mkdir -p log

python geofno_train.py \
    --k_max 12 \
    --bsz 8 \
    --ep 200 \
    --n_train 400 \
    --n_test 80 \
    --layer_sizes 64,64,64,64 \
    --model_name "model/geofno" \
    > log/geofno.log
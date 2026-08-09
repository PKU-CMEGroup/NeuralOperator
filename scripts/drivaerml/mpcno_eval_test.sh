#!/bin/bash
#SBATCH -o out/MPCNO_eval_test.out
#SBATCH --qos=low
#SBATCH -J MPCNO_eval
#SBATCH -p GPU80G
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

mkdir -p log out model

python mpcno_eval_test.py \
    --model_path "model/mpcno.pth" \
    --data_path "../../data/hifi3d_processed/test/drivaerml_vertex_centered.npz" \
    --names_path "../../data/hifi3d_processed/test/drivaerml_names.npy" \
    --raw_data_dir "../../data/HiFi3D/DrivAerML_20000" \
    --output_csv "model/mpcno_test_errors.csv" \
    --seed 0 \
    --n_train 400 \
    --n_test 80 \
    --grad True \
    --geo True \
    --geointegral True \
    --k_max 12 \
    --layer_sizes 64,64,64,64 \
    --batch_size 1 \
    > log/mpcno_eval_test.log

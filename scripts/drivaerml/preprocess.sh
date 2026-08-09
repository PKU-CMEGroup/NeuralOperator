#!/bin/bash
#SBATCH -o drivaerml_preprocess.out
#SBATCH --qos=low
#SBATCH -p C064M0256G
#SBATCH -J drivaerml_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00

source ~/.bashrc
conda activate myconda

mkdir -p log

python preprocess_data.py \
                --data_root ../../data/HiFi3D/ \
                --datasets DrivAerML \
                --n_each 0 \
                --seed 0 \
                --output_dir ../../data/hifi3d_processed/test \
                --output_name drivaerml \
                --mesh_type vertex_centered \
                --adjacent_type edge \
                > log/preprocess_DrivAerML.log
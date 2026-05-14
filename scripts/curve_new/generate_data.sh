#!/bin/bash
#SBATCH -o out/data_generate.out 
#SBATCH --qos=low
#SBATCH -J data_generate
#SBATCH -p C064M1024G
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

source ~/.bashrc
conda activate myconda
python generate_curve_data_panel.py \
    --n_data 10000\
    --beta_low 0.5 \
    --beta_high 1.0 \
    --kernel_type "fourier_param" \
    --fourier_kmax 16 \
    # --two_curves \
#!/bin/bash
#SBATCH -o data_generate.out 
#SBATCH --qos=low
#SBATCH -J data_generate
#SBATCH -p C064M0256G
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

source ~/.bashrc
conda activate myconda
python generate_curves_data_panel.py
#!/bin/bash
#SBATCH -o gen_data.out
#SBATCH --qos=low
#SBATCH -J gen_data
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

module load conda
source ~/.bashrc
conda activate pytorch 

python gen_KS_data.py 

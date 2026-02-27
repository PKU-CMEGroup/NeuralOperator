#!/bin/bash
#SBATCH -o cpu.out
#SBATCH --qos=low
#SBATCH -J cpu
#SBATCH --nodes=1 
#SBATCH --ntasks=2
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python generate_schrodinger1d_data.py

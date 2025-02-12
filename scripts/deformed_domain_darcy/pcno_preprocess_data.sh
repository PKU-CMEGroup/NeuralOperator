#!/bin/bash
#SBATCH -o PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

module load conda
conda activate pytorch 
python darcy_train.py "preprocess_data"

#!/bin/bash
#SBATCH -o logs/PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -p C064M1024G
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python pcno_mixed_3d_test.py --preprocess_data True

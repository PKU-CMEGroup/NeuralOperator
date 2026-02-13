#!/bin/bash
#SBATCH -o PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python pcno_semigroup_burgers_test.py "preprocess_data"

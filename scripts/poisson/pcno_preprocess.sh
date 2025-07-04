#!/bin/bash
#SBATCH -o PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python pcno_test.py --problem_type "preprocess_data"


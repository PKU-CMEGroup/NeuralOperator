#!/bin/bash
#SBATCH -o PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --time=10:00:00

module load conda
source activate pytorch 
python mitral_valve_test.py "preprocess_data"

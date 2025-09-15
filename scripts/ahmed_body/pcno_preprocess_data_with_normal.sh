#!/bin/bash
#SBATCH -o out/PCNO_preprocess_data.out
#SBATCH --qos=low
#SBATCH -J PCNO_preprocess_data
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=100:00:00
source activate o3d
python pcno_ahmedbody_test.py "preprocess_data"

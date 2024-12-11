#!/bin/bash
#SBATCH -o out/job.%j.out
#SBATCH -p GPU80G
#SBATCH --qos=low
#SBATCH -J car_pressure
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1


source activate o3d
python geokno_car_pressure_test.py > log/test.log
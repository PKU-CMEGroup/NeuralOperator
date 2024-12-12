#!/bin/bash
#SBATCH -o PiT_train.out
#SBATCH --qos=low
#SBATCH -J PiT_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python pit_darcy_test.py > PiT_darcy.log


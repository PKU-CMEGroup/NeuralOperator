#!/bin/bash
#SBATCH -o PCNO_train.out
#SBATCH --qos=low
#SBATCH -J Mitral_Valve
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python mitral_valve_test.py --train_sp_L 'together'> PCNO_mitral_valve_test.log


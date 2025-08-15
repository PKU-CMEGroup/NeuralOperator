#!/bin/bash
#SBATCH -o PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 
python semigroup_pcno_adv_test.py --train_inv_L_scale 'independently' > logs/semigroup_pcno_adv_2048_indep.log

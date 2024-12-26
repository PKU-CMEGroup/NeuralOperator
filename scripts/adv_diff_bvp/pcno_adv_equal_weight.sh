#!/bin/bash
#SBATCH -o out/PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_adv_1000_equal_weight
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


source activate pytorch

python pcno_adv_test_equal_weight.py --train_type 'mixed' --n_train 1000 --train_sp_L 'False'> log/PCNO_adv_1000_equal_weight/mixed_False.log
python pcno_adv_test_equal_weight.py --train_type 'mixed' --n_train 1000 --train_sp_L 'independently'> log/PCNO_adv_1000_equal_weight/mixed_independently.log
python pcno_adv_test_equal_weight.py --train_type 'mixed' --n_train 1000 --train_sp_L 'together'> log/PCNO_adv_1000_equal_weight/mixed_together.log
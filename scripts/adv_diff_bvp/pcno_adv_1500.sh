#!/bin/bash
#SBATCH -o out/PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_adv_1500
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


source activate pytorch
python pcno_adv_test.py --train_type 'uniform' --n_train 1500 --train_sp_L 'False' > log/PCNO_adv_1500/uniform_False.log
python pcno_adv_test.py --train_type 'exponential' --n_train 1500 --train_sp_L 'False'> log/PCNO_adv_1500/exponential_False.log
python pcno_adv_test.py --train_type 'linear' --n_train 1500 --train_sp_L 'False'> log/PCNO_adv_1500/linear_False.log
python pcno_adv_test.py --train_type 'mixed' --n_train 1500 --train_sp_L 'False'> log/PCNO_adv_1500/mixed_False.log

python pcno_adv_test.py --train_type 'uniform' --n_train 1500 --train_sp_L 'independently' > log/PCNO_adv_1500/uniform_independently.log
python pcno_adv_test.py --train_type 'exponential' --n_train 1500 --train_sp_L 'independently'> log/PCNO_adv_1500/exponential_independently.log
python pcno_adv_test.py --train_type 'linear' --n_train 1500 --train_sp_L 'independently'> log/PCNO_adv_1500/linear_independently.log
python pcno_adv_test.py --train_type 'mixed' --n_train 1500 --train_sp_L 'independently'> log/PCNO_adv_1500/mixed_independently.log

python pcno_adv_test.py --train_type 'uniform' --n_train 1500 --train_sp_L 'together' > log/PCNO_adv_1500/uniform_together.log
python pcno_adv_test.py --train_type 'exponential' --n_train 1500 --train_sp_L 'together'> log/PCNO_adv_1500/exponential_together.log
python pcno_adv_test.py --train_type 'linear' --n_train 1500 --train_sp_L 'together'> log/PCNO_adv_1500/linear_together.log
python pcno_adv_test.py --train_type 'mixed' --n_train 1500 --train_sp_L 'together'> log/PCNO_adv_1500/mixed_together.log
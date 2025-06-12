#!/bin/bash
#SBATCH -o out/PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_adv_1000
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00


source activate pytorch
python pcno_adv_test.py --train_distribution 'uniform' --n_train 1000 --train_inv_L_scale 'False' --equal_weight 'False' > log/PCNO_adv_1000/uniform_False.log
python pcno_adv_test.py --train_distribution 'exponential' --n_train 1000 --train_inv_L_scale 'False' --equal_weight 'False' > log/PCNO_adv_1000/exponential_False.log
python pcno_adv_test.py --train_distribution 'linear' --n_train 1000 --train_inv_L_scale 'False' --equal_weight 'False' > log/PCNO_adv_1000/linear_False.log
python pcno_adv_test.py --train_distribution 'mixed' --n_train 1000 --train_inv_L_scale 'False' --equal_weight 'False' > log/PCNO_adv_1000/mixed_False.log

python pcno_adv_test.py --train_distribution 'uniform' --n_train 1000 --train_inv_L_scale 'independently' --equal_weight 'False' > log/PCNO_adv_1000/uniform_independently.log
python pcno_adv_test.py --train_distribution 'exponential' --n_train 1000 --train_inv_L_scale 'independently' --equal_weight 'False' > log/PCNO_adv_1000/exponential_independently.log
python pcno_adv_test.py --train_distribution 'linear' --n_train 1000 --train_inv_L_scale 'independently' --equal_weight 'False' > log/PCNO_adv_1000/linear_independently.log
python pcno_adv_test.py --train_distribution 'mixed' --n_train 1000 --train_inv_L_scale 'independently' --equal_weight 'False' > log/PCNO_adv_1000/mixed_independently.log

python pcno_adv_test.py --train_distribution 'uniform' --n_train 1000 --train_inv_L_scale 'together' --equal_weight 'False' > log/PCNO_adv_1000/uniform_together.log
python pcno_adv_test.py --train_distribution 'exponential' --n_train 1000 --train_inv_L_scale 'together' --equal_weight 'False' > log/PCNO_adv_1000/exponential_together.log
python pcno_adv_test.py --train_distribution 'linear' --n_train 1000 --train_inv_L_scale 'together' --equal_weight 'False' > log/PCNO_adv_1000/linear_together.log
python pcno_adv_test.py --train_distribution 'mixed' --n_train 1000 --train_inv_L_scale 'together' --equal_weight 'False' > log/PCNO_adv_1000/mixed_together.log
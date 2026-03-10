#!/bin/bash
#SBATCH -o gpu.out
#SBATCH --qos=low

#SBATCH -J gpu
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1

# SBATCH -J cpu
# SBATCH --nodes=1 
# SBATCH --ntasks=2

#SBATCH --time=10:00:00

module load conda
source activate pytorch 

# python fno_train.py > FNO_train.log
# python fno_plot_results.py  > FNO_test.log

python pcno_train.py > PCNO_train.log
python pcno_plot_results.py  > PCNO_test.log

python pcno_periodic_train.py > PCNO_periodic_train.log
python pcno_periodic_plot_results.py  > PCNO_periodic_test.log

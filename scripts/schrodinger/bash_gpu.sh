#!/bin/bash
#SBATCH -o gpu.out
#SBATCH --qos=low
#SBATCH -J gpu
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch 

python transformer_train.py > TRANSFORMER_train.log
python transformer_plot_results.py  > TRANSFORMER_test.log

python fno_train.py > FNO_train.log
python fno_plot_results.py  > FNO_test.log





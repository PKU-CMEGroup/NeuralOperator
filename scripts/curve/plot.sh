#!/bin/bash
#SBATCH -o PCNO_train.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source ~/.bashrc
conda activate myconda
python freq_analy_singlelayer.py > PCNO_freq.txt
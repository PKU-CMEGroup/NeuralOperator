#!/bin/bash
#SBATCH -o out/PCNO_train_log.out
#SBATCH --qos=low
#SBATCH -J PCNO_train
#SBATCH -p GPU40G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source activate pytorch 
python pcno_curve_geo_test2.py --grad True --geo False --lap False --k_max 8 > log/1_1_5_2d_log_panel/k8_L10_layer2_noact_new.log
python pcno_curve_geo_test2.py --grad True --geo True --lap False --k_max 8  > log/1_1_5_2d_log_panel/k8_L10_layer2_noact_geo3wx_new.log
python pcno_curve_geo_test2.py --grad False --geo True --lap False --k_max 8  > log/1_1_5_2d_log_panel/k8_L10_layer2_noact_nograd_geo3wx_new.log
python pcno_curve_geo_test2.py --grad False --geo False --lap False --k_max 8  > log/1_1_5_2d_log_panel/k8_L10_layer2_noact_nograd_new.log
# python pcno_curve_geo_test.py --grad True --geo False --lap False  --normal_prod True > log/1_1_5_2d_log_panel/k16_L10_layer2_noact_np.log
# python pcno_curve_geo_test.py --grad True --geo True --lap False  --normal_prod True > log/1_1_5_2d_log_panel/k16_L10_layer2_noact_np_geo3wx.log
# python pcno_curve_geo_test.py --grad False --geo True --lap False  --normal_prod True > log/1_1_5_2d_log_panel/k16_L10_layer2_noact_np_nograd_geo3wx.log
# python pcno_curve_geo_test.py --grad False --geo False --lap False  --normal_prod True > log/1_1_5_2d_log_panel/k16_L10_layer2_noact_np_nograd.log


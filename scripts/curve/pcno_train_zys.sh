
python pcno_curve_geo_test.py --kernel_type dp_laplace --grad True --geo True --geointegral False \
 --act gelu --normal_prod True --n_train 2000 --n_test 1000 --n_two_circles_test 1000 --layer_sizes 128,128,128,128,128 --bsz 32 \
 > log/1_1_5_2d_dp_laplace_panel/gelu/n2000_k16_L10_np_grad_geo3.log

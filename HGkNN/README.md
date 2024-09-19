# HGkNN修改的部分：
models/HGalerkin.py  
models/train.py  
models/\_\_init\_\_.py  

# Darcy_HGkNN.py或airfoil_HGkNN.py中的一些修改:
bases_list的含义： \[bases1,wbases1,bases2,wbases2\],在不启用double_bases时只会用到前两个，用wbases1算系数，再用bases1返回物理空间.  
airfoil_HGkNN.py中的pca bases是在原来的数据经过normalization encode以后再做pca得到的  

# config.yml中若干参数的意义：
double_bases: 是否把每个splayer换为两个由不同基组合而成的splayer的和  
regularization_ep: 开始使用正则化的epochs数，一般设置成0，表示开始训练即启用正则化（其他关于正则化的系数全为0或未写入时，不会启用正则化）  
L1regularization_lambda: 对H使用L1正则化的lambda数  
hard_pruning: 对H中小于该数的部分强行修改为0  
grad_clip： 梯度裁剪  
L2regularization_alpha： 对splayers中的训练weight运用的L2正则化  
L2regularization_beta： 对H运用的L2正则化  
L2regularization_symmetry： 对H-H^T运用的L2正则化

plot_H_num: 在训练过程中画出的H的个数  
plot_hidden_layers： 在训练过程中是否画出中间层的图像  
save_figure_H： 保存画出的H的地址  （如果plot_H_num>0，那么这一项的地址对应的文件夹必须事先存在）  
save_figure_hidden： 保存画出的中间层图像的地址  （如果plot_hidden_layers为True，那么这一项的地址对应的文件夹必须事先存在）  
plot_shape： 每个问题对应的图像shape
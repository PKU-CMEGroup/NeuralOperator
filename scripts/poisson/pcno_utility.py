import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
from itertools import accumulate
from scipy.spatial import cKDTree
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def compute_min_point_distance(boundary, query):
    tree = cKDTree(boundary)
    distances, _ = tree.query(query, k=1)
    return distances

# 从 data_path 读入 ndata 数据
# 每一个文件包含 2 个数据，分别对应于高频、低频的边界函数值
# 因此读入ndata数据，只需要读入ndata//2文件
# 输出
# nodes_list ：node 坐标
# elems_list ：elment 信息， [elem_dim, e1 e2...], padding with -1
# features_list ：Laplace solution, Poisson solution, source term, boundary condition, boundary node indicator, SDF, 
def load_raw_data(data_path, ndata):
    assert(ndata%2 == 0)
    path_all = data_path + "/All"
    path_boundary = data_path + "/Boundary"

    nodes_list = []
    elems_list = []
    features_list = []

    for i in range(ndata//2):
        # node coordinates
        nodes = np.load(path_all + "/nodes_%05d" % (i) + ".npy")
        # triangular elements
        telems = np.load(path_all + "/elems_%05d" % (i) + ".npy")
        # boundary line elements
        belems = np.load(path_boundary + "/elems_%05d" % (i) + ".npy")
        # element list [elem_dim, e1 e2...], padding with -1
        elems = np.vstack((np.concatenate((np.full((telems.shape[0], 1), 2, dtype=int), telems), axis=1),
                                     np.concatenate((np.full((belems.shape[0], 1), 1, dtype=int), belems, np.full((belems.shape[0], 1), -1, dtype=int)), axis=1)
                                   ))
        # number of nodes
        nnodes = nodes.shape[0]
        # features for all nodes from data: 
        # 0:laplace solution for low-freq bc, 1:laplace solution for high-freq bc, 2:source, 3:zero-dirichlet solution, 4:poisson solution for low-freq bc, 5:poisson solution for high-freq bc, ...
        features = np.load(path_all + "/features_%05d" % (i) + ".npy")
        # boundary node indices
        bnodes = np.unique(belems)
        # features for boundary nodes from data
        # low-freq bc, high-freq bc, ...
        bfeatures_only = np.load(path_boundary + "/features_%05d" % (i) + ".npy")
        # boundary related features
        bfeatures = np.zeros((nnodes, bfeatures_only.shape[1] + 2))
        bfeatures[bnodes, 0:-2] = bfeatures_only
        # indicator function for boundary nodes
        bfeatures[bnodes, -2] = 1
        # distance function for all nodes
        bfeatures[:, -1] = compute_min_point_distance(nodes[bnodes,:], nodes)
        
        # low freq boundary condition
        nodes_list.append(nodes) 
        elems_list.append(elems)   
        features_list.append(np.hstack((features[:, [0,4,2]], bfeatures[:,[0,-2,-1]])))
        # high freq boundary condition
        nodes_list.append(nodes) 
        elems_list.append(elems)   
        features_list.append(np.hstack((features[:, [1,5,2]], bfeatures[:,[1,-2,-1]])))
        
    return nodes_list, elems_list, features_list






def get_median_index(arr):
    # 确保输入是一个 NumPy 数组
    arr = np.asarray(arr)
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 计算中位数的索引
    mid_index = len(arr) // 2
    
    if len(arr) % 2 == 1:
        # 如果是奇数长度，返回中间元素的原始索引
        median_index = sorted_indices[mid_index]
    else:
        # 如果是偶数长度，返回中间两个元素的原始索引
        median_index_1 = sorted_indices[mid_index - 1]
        median_index_2 = sorted_indices[mid_index]
        # 通常我们不会为偶数长度的数组返回单个索引，因为中位数是两个值的平均。
        # 但是，如果你需要，你可以选择返回这两个索引或仅其中一个。
        # 这里我们简单地返回一个元组
        median_index = median_index_1
    
    return median_index



def plot_colorbar(fig, c, ax_list):
    cbar = fig.colorbar(c, ax=ax_list)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(direction="in")

# 从 data_path_pref+shape 读入第i个数据
# 每一个文件包含 2 个数据，分别对应于高频、低频的边界函数值
# 因此读入第i个数据，只需要读入第i//2个文件
def plot_solution(problem_type, data_path_pref, shape, i, sol_pred, ax, fig):
    idx, f_idx = i//2, i%2
    elems = np.load(data_path_pref + shape + "/All/elems_%05d" % (idx) + ".npy")
    nodes = np.load(data_path_pref + shape + "/All/nodes_%05d" % (idx) + ".npy")
    features = np.load(data_path_pref + shape + "/All/features_%05d" % (idx) + ".npy")
    belems = np.load(data_path_pref + shape + "/Boundary/elems_%05d" % (idx) + ".npy")
    bfeatures = np.load(data_path_pref + shape + "/Boundary/features_%05d" % (idx) + ".npy")

    segments = [[nodes[i], nodes[j]] for i, j in belems]
    values_per_edge = np.array([(bfeatures[i, f_idx] + bfeatures[j, f_idx]) / 2 for i, j in belems]).reshape(-1)
    g_min, g_max = np.min(bfeatures[:, f_idx]), np.max(bfeatures[:, f_idx])
    
    ax[0].triplot(nodes[:, 0], nodes[:, 1], elems, linewidth=0.15, alpha=0.6)

    lc = LineCollection(segments, array=values_per_edge,
                        linewidths=3, norm=plt.Normalize(vmin=g_min, vmax=g_max))
    ax[1].add_collection(lc)
    ax[1].autoscale()
    plot_colorbar(fig, lc, ax[1])
    # Plot source term
    if problem_type == "Poisson":
        lc = ax[2].tripcolor(nodes[:, 0], nodes[:, 1], elems, features[:, 2])
        plot_colorbar(fig, lc, ax[2])

    v_min, v_max = np.min(features[:, f_idx]), np.max(features[:, f_idx])
    lc = ax[-3].tripcolor(nodes[:, 0], nodes[:, 1], elems, features[:, f_idx], vmin=v_min, vmax=v_max)
    plot_colorbar(fig, lc, ax[-3])
    
    lc = ax[-2].tripcolor(nodes[:, 0], nodes[:, 1], elems, sol_pred, vmin=v_min, vmax=v_max)
    plot_colorbar(fig, lc, ax[-2])
    
    lc = ax[-1].tripcolor(nodes[:, 0], nodes[:, 1], elems, np.fabs(features[:, f_idx] - sol_pred))
    plot_colorbar(fig, lc, ax[-1])
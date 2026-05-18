import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.mpcno_beta_lowrank_dyn import compute_Fourier_modes, MPCNO_Beta, MPCNO_train_multidist_beta

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# 参数解析
###################################

parser = argparse.ArgumentParser(
    description='Train MPCNO_Beta (Tucker hyper-net + FiLM) on anisotropic sp_laplace data.')

parser.add_argument("--kernel_type", type=str, default="sp_laplace", choices=["sp_laplace","dp_laplace"])
parser.add_argument('--grad',              type=str,   default='True',  choices=['True', 'False'])
parser.add_argument('--geo',               type=str,   default='True',  choices=['True', 'False'])
parser.add_argument('--geointegral',       type=str,   default='True',  choices=['True', 'False'])
parser.add_argument('--to_divide_factor',  type=float, default=20.0)
parser.add_argument('--k_max',             type=int,   default=16)
parser.add_argument('--bsz',               type=int,   default=32)
parser.add_argument('--ep',                type=int,   default=500)
parser.add_argument('--n_train',           type=int,   default=2000)
parser.add_argument('--n_test',            type=int,   default=1000)
# Mirror of --n_two_circles_test in original mpcno_curve_test.py
parser.add_argument('--n_two_circles_test', type=int,   default=0,
                    help='Number of two-curves samples to evaluate as a second test set (0 = skip)')
parser.add_argument('--act',               type=str,   default='gelu')
parser.add_argument('--geo_act',           type=str,   default='softsign')
parser.add_argument('--layer_sizes',       type=str,   default='64,64,64,64,64,64')
parser.add_argument('--hyper_hidden', type=int, default=64,
                    help='Hidden width of hyper-MLPs for DynLowRankSpectralConv (default 64)')
parser.add_argument('--rank', type=int, default=8,
                    help='Low rank r for DynLowRankSpectralConv (default 8)')
parser.add_argument('--film_hidden', type=int, default=32,
                    help='Hidden width of FiLM generators for Geo/Grad layers (default 32)')
# 新增 beta 范围参数
parser.add_argument('--beta_low', type=float, default=0.5)
parser.add_argument('--beta_high', type=float, default=2.5)
parser.add_argument('--beta_dim', type=int, default=1)

args = parser.parse_args()

layer_selection = {
    'grad':        args.grad.lower()        == 'true',
    'geo':         args.geo.lower()         == 'true',
    'geointegral': args.geointegral.lower() == 'true',
}

# sp_laplace with beta: scalar f in, scalar g out
f_in_dim  = 1
f_out_dim = 1

k_max             = args.k_max
ndim              = 2
L                 = 10
layers            = [int(s) for s in args.layer_sizes.split(',')]
act               = args.act
geo_act           = args.geo_act
beta_dim          = args.beta_dim
hyper_hidden      = args.hyper_hidden
rank              = args.rank
film_hidden       = args.film_hidden
to_divide_factor  = args.to_divide_factor
train_inv_L_scale = False

###################################
# 数据加载工具
###################################

def load_data_to_torch(data_file_path, to_divide=None, factor=1.0):
    """
    与原始 mpcno_curve_test.py 的 load_data_to_torch 相同，
    额外读取 betas 字段。

    Returns:
        nnodes, node_mask, nodes, node_weights, node_rhos,
        features, directed_edges, edge_gradient_weights,
        betas  [ndata,],
        to_divide
    """
    data = np.load(data_file_path)
    nnodes            = data['nnodes']
    node_mask         = data['node_mask']
    nodes             = data['nodes']
    node_measures_raw = data['node_measures_raw']

    print(f'Loaded {nodes.shape[0]} samples from {data_file_path}', flush=True)

    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_measures_raw, axis=1))
    node_weights = node_measures_raw / to_divide

    node_measures         = data['node_measures']
    directed_edges        = data['directed_edges']
    edge_gradient_weights = data['edge_gradient_weights']
    features              = data['features']

    indices   = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    betas = data['betas'].astype(np.float32)   # [ndata,]

    nnodes                = torch.from_numpy(nnodes)
    node_mask             = torch.from_numpy(node_mask)
    nodes                 = torch.from_numpy(nodes.astype(np.float32))
    node_weights          = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos             = torch.from_numpy(node_rhos.astype(np.float32))
    features              = torch.from_numpy(features.astype(np.float32))
    directed_edges        = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
    betas                 = torch.from_numpy(betas)

    return (nnodes, node_mask, nodes, node_weights, node_rhos,
            features, directed_edges, edge_gradient_weights,
            betas, to_divide)


def gen_data_tensors(data_indices,
                     nodes, features, node_mask, node_weights,
                     directed_edges, edge_gradient_weights, betas):
    """
    x layout : [f (1), normal_x, normal_y, coord_x, coord_y]  ->  5 dims
    y layout : [g (1)]
    beta      : [B, 1]
    """
    nodes_input = nodes.clone()

    x = torch.cat([
        features[data_indices][..., :f_in_dim + 2],   # f + outward normals
        nodes_input[data_indices],                    # coordinates
    ], dim=-1)

    y   = features[data_indices][..., -f_out_dim:]
    nx  = features[data_indices][..., f_in_dim:f_in_dim + 2]
    aux = (
        node_mask[data_indices],
        nodes[data_indices],
        node_weights[data_indices],
        directed_edges[data_indices],
        edge_gradient_weights[data_indices],
        nx.permute(0, 2, 1),    # [B, 2, N]
    )
    beta_batch = betas[data_indices].unsqueeze(-1)   # [B, 1]

    return x, y, beta_batch, aux


###################################
# 加载 single-curve 数据（训练集 + 默认测试集）
###################################

data_path = '../../data/curve_beta/'

(nnodes, node_mask, nodes, node_weights, node_rhos,
 features, directed_edges, edge_gradient_weights,
 betas, to_divide) = load_data_to_torch(
    data_path + f"/pcno_curve_data_1_1_5_beta({args.beta_low}, {args.beta_high})_2d_{args.kernel_type}_beta_random_panel_single.npz",
    to_divide=None, factor=to_divide_factor)

n_train      = args.n_train
n_test       = args.n_test
n_two_curves = args.n_two_circles_test

x_train, y_train, beta_train, aux_train = gen_data_tensors(
    np.arange(n_train),
    nodes, features, node_mask, node_weights,
    directed_edges, edge_gradient_weights, betas)

x_test, y_test, beta_test, aux_test = gen_data_tensors(
    np.arange(-n_test, 0),
    nodes, features, node_mask, node_weights,
    directed_edges, edge_gradient_weights, betas)

x_test_list    = [x_test]
y_test_list    = [y_test]
aux_test_list  = [aux_test]
beta_test_list = [beta_test]
label_list     = ['Single']

###################################
# （可选）加载 two-curves 测试集
# 与原版 mpcno_curve_test.py 的 n_two_circles_test 逻辑完全对称
###################################

if n_two_curves > 0:
    (nnodes2, node_mask2, nodes2, node_weights2, node_rhos2,
     features2, directed_edges2, edge_gradient_weights2,
     betas2, _) = load_data_to_torch(
        data_path + f"/pcno_curve_data_1_1_5_beta({args.beta_low}, {args.beta_high})_2d_{args.kernel_type}_beta_random_panel_two_curves.npz",
        to_divide=to_divide)   # 与 single 共享同一 to_divide

    x_two, y_two, beta_two, aux_two = gen_data_tensors(
        np.arange(n_two_curves),
        nodes2, features2, node_mask2, node_weights2,
        directed_edges2, edge_gradient_weights2, betas2)

    x_test_list.append(x_two)
    y_test_list.append(y_two)
    aux_test_list.append(aux_two)
    beta_test_list.append(beta_two)
    label_list.append('Two Curves')

###################################
# 打印数据信息
###################################

print(f'x_train {x_train.shape}  y_train {y_train.shape}  beta_train {beta_train.shape}',
      flush=True)
print(f'Test sets: { {lbl: x.shape for lbl, x in zip(label_list, x_test_list)} }',
      flush=True)
print(f'beta range in train: [{beta_train.min().item():.4f}, {beta_train.max().item():.4f}]')
print(f'Domain range per dim: {torch.amax(nodes, dim=[0,1]) - torch.amin(nodes, dim=[0,1])}',
      flush=True)

###################################
# 模型初始化
###################################

modes = compute_Fourier_modes(ndim, [k_max, k_max], [L, L])
modes = torch.tensor(modes, dtype=torch.float).to(device)

print('\n------Parameters------')
print(f'k_max = {k_max},  L = {L}')
print(f'n_train = {n_train},  n_test = {n_test},  n_two_curves_test = {n_two_curves}')
print(f'Fourier modes shape: {modes.shape}')
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')
print(f'beta_dim = {beta_dim},  hyper_hidden = {hyper_hidden},  rank = {rank},  film_hidden = {film_hidden}')
print(f'activation = {act},  geo_activation = {geo_act}')

model = MPCNO_Beta(
    ndims           = ndim,
    modes           = modes,
    nmeasures       = 1,
    layers          = layers,
    beta_dim        = beta_dim,
    hyper_hidden    = hyper_hidden,
    rank            = rank,
    film_hidden     = film_hidden,
    layer_selection = layer_selection,
    fc_dim          = 128,
    in_dim          = x_train.shape[-1],   # 5: f + normals + coords
    out_dim         = y_train.shape[-1],   # 1: g
    inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
    scaling_mode    = 'inv',
    act             = act,
    geo_act         = geo_act,
).to(device)

n_params        = sum(p.numel() for p in model.parameters())
# 新版 DynLowRankSpectralConv：没有静态因子，只有 hyper_net + 基础权重
n_hyper_params = sum(p.numel()
                     for sc in model.sp_convs
                     for p in list(sc.hyper_net_m.parameters())
                            + [sc.weights_c, sc.weights_s, sc.weights_0])

n_film_params = 0
if layer_selection['geo']:
    n_film_params += sum(p.numel()
                         for ge in model.geo_embs
                         for p in ge.film_net.parameters())
if layer_selection['grad']:
    n_film_params += sum(p.numel()
                         for gl in model.grad_layers
                         for p in gl.film_net.parameters())
print(f'Total parameters:                    {n_params:,}')
print(f'  DynLowRank hyper (SpectralConv):   {n_hyper_params:,}  ({100*n_hyper_params/n_params:.2f}%)')
print(f'  FiLM (Geo + Grad):                 {n_film_params:,}  ({100*n_film_params/n_params:.2f}%)')
###################################
# 训练配置
###################################

batch_size = args.bsz
print('batch_size', batch_size, '\n')

config = {
    "train": {
        "base_lr":              5e-4,
        "lr_ratio":             10,
        "weight_decay":         1e-4,
        "epochs":               args.ep,
        "scheduler":            "OneCycleLR",
        "batch_size":           batch_size,
        "normalization_x":      False,
        "normalization_y":      True,
        "normalization_dim_x":  [],
        "normalization_dim_y":  [],
        "non_normalized_dim_x": 4,
        "non_normalized_dim_y": 0,
    }
}

###################################
# 训练
###################################

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = \
    MPCNO_train_multidist_beta(
        x_train, aux_train, y_train, beta_train,
        x_test_list, aux_test_list, y_test_list, beta_test_list,
        config, model,
        label_test_list=label_list,
        save_model_name=None,
    )
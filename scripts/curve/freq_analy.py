import os
import torch

import sys

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent.parent))

import numpy as np
from timeit import default_timer

# 替换原来的问题代码
current_dir = Path(os.getcwd())  # 获取当前工作目录
print(current_dir)
project_root = current_dir.parent  # 上两级目录
print(project_root)
sys.path.insert(0, str(project_root))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train
from modes_discrete import discrete_half_ball_modes, nonlinear_scale
from modes_discrete_zys import discrete_half_ball_modes as zys_discrete_half_ball_modes
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load data 
equal_weights = False
data_path = ""
data = np.load(data_path+"preprocessed/2D_neblalog_data_3_3_10000.npz")
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
normals = data["normals"]
print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
# node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_weights = data["node_measures_raw"]
# print('use node_weight')
node_weights = node_weights/np.amax(np.sum(node_weights, axis = 1))
print('use normalized raw measures')
node_measures = data["node_measures"]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]

node_measures_raw = data["node_measures_raw"]
indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_weights)
node_rhos[indices] = node_rhos[indices]/node_measures[indices]

print("Casting to tensor",flush = True)
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
normals = torch.from_numpy(normals.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


nodes_input = nodes.clone()

n_train, n_test = 9000, 1000


fnormals = normals * features[..., :1]
x_train, x_test = torch.cat((features[:n_train, :, :1], nodes_input[:n_train, ...], node_rhos[:n_train, ...], fnormals[:n_train, ...]), -1), torch.cat((features[-n_test:, :, :1],nodes_input[-n_test:, ...], node_rhos[-n_test:, ...], fnormals[-n_test:, ...]),-1)

aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

y_train, y_test = features[:n_train, :, 1:],     features[-n_test:, :, 1:]

print(f'x_train shape {x_train.shape}, y_train shape {y_train.shape}')
print('length of each dim: ',torch.amax(nodes_input, dim = [0,1]) - torch.amin(nodes_input, dim = [0,1]), flush = True)
train_inv_L_scale = False
k_max = 8 
scale = 0
min_dir_fraction = 0.3  
print(f'kmax = {k_max}, scale = {scale}, min_dir_fraction = {min_dir_fraction}')
ndim = 2
L = 6.0
print('use box size L=', L)

'''
modes = zys_discrete_half_ball_modes(ndim, k_max, scale) * k_max * 2 * np.pi / L
modes = modes[:,:,np.newaxis]
modes = nonlinear_scale(modes, scale=scale)
print('use sphere modes', modes.shape)
'''
modes = compute_Fourier_modes(ndim, [k_max,k_max], [L,L])
modes = nonlinear_scale(modes, scale=scale)
print("use cube modes", modes.shape)

modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act='gelu').to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = False
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 4
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }


model_name="model/False万1.pth"
model.load_state_dict(torch.load(model_name))
model.to(device)
print("Model loaded from ", model_name)
model.eval()
from mpl_toolkits.mplot3d import Axes3D

weights_c = model.sp_convs[0].weights_c  # weight_c: [in_channels, out_channels, nmodes, nmeasures]
weights_s = model.sp_convs[0].weights_s  # weight_s: [in_channels, out_channels, nmodes, nmeasures]
weights_0 = model.sp_convs[0].weights_0

print(weights_c.shape,weights_s.shape,weights_0.shape)

modes = modes.cpu().numpy()
weights_c = weights_c.detach().cpu().numpy()
weights_s = weights_s.detach().cpu().numpy()
weights_0 = weights_0.detach().cpu().numpy()

def reconstruct(modes: np.ndarray, coeffs, X: np.ndarray):
    """
    Real trig reconstruction:
        f(x) = c0 + Σ_k [ a_k cos(m_k·x) + b_k sin(m_k·x) ]
    coeffs can be:
      1) flattened array: [c0, a0, b0, a1, b1, ...]
      2) dict with keys: c0, coeffs_cos, coeffs_sin
    """
    modes = np.asarray(modes, float)
    X = np.asarray(X, float)
    M = modes.shape[0]

    if isinstance(coeffs, dict):
        c0 = float(coeffs["c0"])
        a = np.asarray(coeffs["coeffs_cos"], float)
        b = np.asarray(coeffs["coeffs_sin"], float)
    else:
        coeffs = np.asarray(coeffs, float)
        expected = 1 + 2 * M
        if coeffs.size != expected:
            raise ValueError(f"coeffs length {coeffs.size} != 1+2*M = {expected}")
        c0 = coeffs[0]
        a = coeffs[1::2][:M]
        b = coeffs[2::2][:M]

    if M == 0:
        return np.full(X.shape[0], c0, dtype=float)

    theta = X @ modes.T  # (N,M)
    print(modes.shape)
    print(theta.shape)
    return c0 + (np.cos(theta) * a + np.sin(theta) * b).sum(axis=1)

def plot_result(modes, res, domain, nx=200, ny=200, show=True):
    """
    可视化: 重构函数 - 一行两列布局
    """
    import matplotlib.pyplot as plt
    (x0, x1), (y0, y1) = domain
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    f_rec = reconstruct(modes, res, XY).reshape(ny, nx)
    vrec = np.max(np.abs(f_rec))
    common_v = vrec

    # 创建一行两列的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 第一个子图：重构结果
    im1 = ax1.imshow(f_rec, extent=[x0, x1, y0, y1], origin='lower',
                     cmap='viridis', vmin=-common_v, vmax=common_v, aspect='auto')
    ax1.set_title("Reconstructed f̂")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 第二个子图：模式散点图
    modes_array = np.asarray(modes, float)
    M, dim = modes_array.shape
    a = np.asarray(res["coeffs_cos"]) if "coeffs_cos" in res else np.asarray(res["coeffs"][1::2][:M])
    b = np.asarray(res["coeffs_sin"]) if "coeffs_sin" in res else np.asarray(res["coeffs"][2::2][:M])
    mag = np.sqrt(a**2 + b**2)
    
    if dim != 2:
        ax2.text(0.5, 0.5, f"Cannot scatter\nmodes (dim={dim} ≠ 2)", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_axis_off()
    else:
        sc = ax2.scatter(modes_array[:, 0], modes_array[:, 1], c=mag, s=16,
                         edgecolors='k', linewidths=0.3)
        ax2.set_title(r"$\sqrt{a_k^2 + b_k^2}$")
        ax2.set_xlabel("m_x")
        ax2.set_ylabel("m_y")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="amplitude")

    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.1)
    
    return fig

in_id, out_id = 0,0
temp = 0
for in_i in range(128):
    for out_i in range(128):
        a = np.linalg.norm(weights_c[in_i, out_i, ...]) + np.linalg.norm(weights_s[in_i, out_i, ...]) + np.abs(weights_0[in_i, out_i])
        print(a.item(),end=" ")
        if temp < a:
            in_id = in_i
            out_id = out_i
            temp = a
    print()
print(in_id, out_id)

coeffs = {"c0":weights_0[in_id, out_id],
          "coeffs_cos":weights_c[in_id, out_id, ...].flatten(),
          "coeffs_sin":weights_s[in_id, out_id, ...].flatten()
          }

modes = modes.squeeze()
domain = [(-5, 5),(-5, 5)]
fig = plot_result(modes, coeffs, domain, show=False)
fig.savefig("1.png")
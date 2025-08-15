import torch
import numpy as np
import os
import sys

sys.path.append('/lustre/home/2200010815/scow/desktops/NeuralOperator')
from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes
from pcno.tpcno import TPCNO


from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = "../../data/adv_diff/"


# Preprocess data
nodes_list, elems_list, features_list = [], [], []
index = 10
ndata, nt = 1, 100
for i in range(ndata):
    index += i
    data = np.load(data_path + "data_uniform_2048/data_%05d"%(index) + ".npy")
    _, nnodes = data.shape
    nodes = np.linspace(-1, 1, nnodes, endpoint=False)
    nodes = nodes[:,np.newaxis]
    elems = np.vstack((np.full(nnodes - 1, 1), np.arange(0, nnodes - 1), np.arange(1, nnodes))).T
    elems = np.append(elems, np.array([1, nnodes - 1, 0]).reshape(1, 3), axis=0)
    for l in range(nt):
        nodes_list.append(nodes)
        elems_list.append(elems)
        t1 = np.random.randint(1, np.min((7, nt + 1 - l)))
        L1 = data[l, :]
        L2 = data[l + t1, :]
        features_list.append(np.vstack((L1, t1 * np.ones((1, nnodes)) / nt, L2)).T)

nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
node_measures, node_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = False)
node_equal_measures, node_equal_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = True)
edge_gradient_weights[:, 0, :] = np.full((ndata * nt, 1), 512)
edge_gradient_weights[:, 1, :] = np.full((ndata * nt, 1), -512)

equal_weights = True
node_weights = node_equal_weights

indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_weights)
node_rhos[indices] = node_rhos[indices] / node_measures[indices]

node_rhos[:, 0, :] = np.full((ndata * nt, 1), 0.5)
node_rhos[:, 2047, :] = np.full((ndata * nt, 1), 0.5)
edge_gradient_weights[:, 0, :] = np.full((ndata * nt, 1), 512)
edge_gradient_weights[:, 1, :] = np.full((ndata * nt, 1), -512)
edge_gradient_weights[:, 4094, :] = np.full((ndata * nt, 1), 512)
edge_gradient_weights[:, 4095, :] = np.full((ndata * nt, 1), -512) # 周期边界条件

print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


# load model
k_max = 48
ndim = 1
modes = compute_Fourier_modes(ndim, [k_max], [2.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)

model = TPCNO(ndim, modes, nmeasures=2,
            layers=[96, 96, 96, 96, 96],
            fc_dim=96,
            in_dim=4, out_dim=1, 
            inv_L_scale_hyper = ['independently', 0.5, 2.0],
            act='gelu',
            grad=True).to(device)

model.load_state_dict(torch.load('models/Semigroup_PCNO_2048_independently.pth', map_location=device, weights_only=False))


# Prepare test data
cases = 1
n_test = nt * cases
nodes_input = nodes.clone()
x_test = torch.cat((features[-n_test:, :, [0]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]), -1)
node_mask_test = node_mask[-n_test:,...]
nodes_test = nodes[-n_test:,...]
node_weights_test = node_weights[-n_test:,...]
directed_edges_test = directed_edges[-n_test:,...]
edge_gradient_weights_test = edge_gradient_weights[-n_test:,...]
t_test = features[-n_test:, :, [1]]
y_test = features[-n_test:, :, [2]]

scale = 1
test_t = scale * nt     # test time steps
pred = np.zeros((test_t, y_test.shape[1], y_test.shape[1]))

x = x_test[[0],...].repeat(test_t, 1, 1).to(device)
t = t_test[[0],...].to(device) / scale
node_mask = node_mask_test[[0],...].to(device)
nodes = nodes_test[[0],...].to(device)
node_weights = node_weights_test[[0],...].to(device)
directed_edges = directed_edges_test[[0],...].to(device)
edge_gradient_weights = edge_gradient_weights_test[[0],...].to(device)

# Predict step by step
for step in tqdm(range(test_t)):
    out = model(x[[step],...], t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights))

    pred[[step], :, :] = out.cpu().detach().numpy()

    if step < test_t - 1:
        x[step + 1 : step + 2, :, [0]] = out

# test_index = np.linspace(scale - 1, test_t - 1, nt).astype(int)
# pred = pred[test_index, :, :]

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

nodes = nodes.cpu().detach().numpy()

line1, = ax.plot(nodes[0, :, 0], data[1, :], label='true', color='r', linewidth=1)
line2, = ax.plot(nodes[0, :, 0], pred[0, :, 0], label='pred', color='b', linewidth=1)
ax.set_title('T = 1')
ax.legend(loc='upper left')

def init():
    line1.set_ydata([np.nan] * len(nodes[0, :, 0]))
    line2.set_ydata([np.nan] * len(nodes[0, :, 0]))
    return line1, line2,

def update(frame):
    line1.set_ydata(data[frame+1, :])
    line2.set_ydata(pred[frame, :, 0])
    ax.set_title(f'T = {round((frame+1)/100, 3)}')
    return line1, line2,

ani = FuncAnimation(fig,
                    update,
                    init_func=init,
                    frames=100,
                    interval=500,
                    blit=True)

ani.save(f'test/data_{str(index).zfill(5)}_{1/scale}.gif', fps=15, writer='imagemagick')



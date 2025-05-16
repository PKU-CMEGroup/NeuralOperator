import torch
import numpy as np
import argparse
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.pcno import compute_Fourier_modes
from bno.geo_utility import mix_data
from bno.gnobno import ExtGNOBNO
from bno.training import BNO_train

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data preprocessing is handled in `bno_poisson_test.py`

print(f"{'%' * 40} Testing ExtGNOBNO on the Poisson Dataset {'%' * 40}", flush=True)
parser = argparse.ArgumentParser(
    description='Train model with different configurations and options.')

parser.add_argument('--m_trains', type=int, nargs='+', default=[512, 512, 512])
parser.add_argument('--shape_types_train', type=str, nargs='+', default=['lowfreq', 'highfreq', 'double'], choices=['lowfreq', 'highfreq', 'double', 'hole'])
parser.add_argument('--m_tests', type=int, nargs='+', default=[128, 128, 128, 128])
parser.add_argument('--shape_types_test', type=str, nargs='+', default=['lowfreq', 'highfreq', 'double', 'hole'], choices=['lowfreq', 'highfreq', 'double', 'hole'])

parser.add_argument('--feature_type_boundary', type=str, nargs=2, default=['Dirichlet', 'g_lowfreq'])
parser.add_argument('--feature_type_all', type=str, default='no_SDF', choices=['no_SDF', 'SDF'])

parser.add_argument('--equal_weights', type=str, default='False')

parser.add_argument('--Lx', type=float, default=2, help='Initial Lx')
parser.add_argument('--Ly', type=float, default=2, help='Initial Ly')
parser.add_argument('--kx_max', type=int, default=16)
parser.add_argument('--ky_max', type=int, default=16)

parser.add_argument('--normalization_x', type=str, default='True')
parser.add_argument('--normalization_y', type=str, default='True')
parser.add_argument('--normalization_out', type=str, default='True')
parser.add_argument('--train_sp_L', type=str, default='independently', choices=['False', 'together', 'independently'])
parser.add_argument('--lr_ratio', type=float, default=10)
parser.add_argument('--checkpoint_path', type=str, default="None")
parser.add_argument('--save_model_ID', type=str, default="None")

args = parser.parse_args()
assert len(args.m_trains) == len(args.shape_types_train)
assert len(args.m_tests) == len(args.shape_types_test)


if args.save_model_ID.lower() == "none":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_setting_str = '_'.join(feature for feature in args.feature_type_boundary)
    train_summary_str = '_'.join(f'{shape}{m}' for shape, m in zip(args.shape_types_train, args.m_trains))
    args.save_model_ID = f"{problem_setting_str}_{train_summary_str}_{current_time}"

for i, (key, value) in enumerate(vars(args).items()):
    if isinstance(value, str):
        if value.lower() == "true":
            setattr(args, key, True)
        elif value.lower() == "false":
            setattr(args, key, False)
        elif value.lower() == "none":
            setattr(args, key, None)
    print(f"{key}: {value}")

m_trains, m_tests = args.m_trains, args.m_tests
n_train, n_test = sum(m_trains), sum(m_tests)

shape_types_train = args.shape_types_train
shape_types_test = args.shape_types_test

boundary_condition, g_freq = args.feature_type_boundary[0], args.feature_type_boundary[1]
if g_freq == "g_lowfreq":
    all_index = [0]
    if boundary_condition == "Dirichlet":
        boundary_index = [0]
    elif boundary_condition == "Neumann":
        boundary_index = [2]
elif g_freq == "g_highfreq":
    all_index = [1]
    if boundary_condition == "Dirichlet":
        boundary_index = [1]
    elif boundary_condition == "Neumann":
        boundary_index = [3]
SDF = True if args.feature_type_all == "SDF" else False
SDF_index = -1

equal_weights = args.equal_weights


######################################################################
# Data
######################################################################
data_folder = "../../data/poisson/preprocessed_data/"

print("\nMixing training data...")
data = mix_data(data_folder, m_trains, shape_types_train, m_tests, shape_types_test)
radius = data['r']
nodes_all, nodes_boundary = data["nodes_all"], data["nodes_boundary"]
mask_all, mask_boundary = data["mask_all"], data["mask_boundary"]
features_all, features_boundary = data["features_all"], data["features_boundary"]
weights_boundary = data["equal_weights_boundary"] if equal_weights else data["weights_boundary"]
rhos_boundary = data["equal_rhos_boundary"] if equal_weights else data["rhos_boundary"]
edges_boundary = data["edges_boundary"]
edgeweights_boundary = data["edgeweights_boundary"]


print("Casting to tensor...")
nodes_x, nodes_y = torch.from_numpy(nodes_all.astype(np.float32)), torch.from_numpy(nodes_boundary.astype(np.float32))
mask_x, mask_y = torch.from_numpy(mask_all), torch.from_numpy(mask_boundary)
weights_y = torch.from_numpy(weights_boundary.astype(np.float32))
rhos_y = torch.from_numpy(rhos_boundary.astype(np.float32))
features_x, features_y = torch.from_numpy(features_all.astype(np.float32)), torch.from_numpy(features_boundary.astype(np.float32))
edges_y = torch.from_numpy(edges_boundary.astype(np.int64))
edgeweights_y = torch.from_numpy(edgeweights_boundary.astype(np.float32))

if SDF:
    x = torch.cat([nodes_x.clone(), features_x[..., [SDF_index]]], dim=-1)
    in_dim_x = 3
else:
    x = nodes_x.clone()
    in_dim_x = 2

y = torch.cat([nodes_y.clone(), features_y[..., boundary_index], rhos_y], dim=-1)
truth = features_x[..., all_index]

x_train, x_test = x[:n_train, ...], x[-n_test:, ...]
y_train, y_test = y[:n_train, ...], y[-n_test:, ...]
truth_train, truth_test = truth[:n_train, ...], truth[-n_test:, ...]
mask_x_train, mask_x_test = mask_x[:n_train, ...], mask_x[-n_test:, ...]
nodes_x_train, nodes_x_test = nodes_x[:n_train, ...], nodes_x[-n_test:, ...]
nodes_y_train, nodes_y_test = nodes_y[:n_train, ...], nodes_y[-n_test:, ...]
weights_y_train, weights_y_test = weights_y[:n_train, ...], weights_y[-n_test:, ...]
edges_y_train, edges_y_test = edges_y[:n_train, ...], edges_y[-n_test:, ...]
edgeweights_y_train, edgeweights_y_test = edgeweights_y[:n_train, ...], edgeweights_y[-n_test:, ...]

aux_train, aux_test = (mask_x_train, nodes_x_train, nodes_y_train, weights_y_train, edges_y_train, edgeweights_y_train), (mask_x_test, nodes_x_test, nodes_y_test, weights_y_test, edges_y_test, edgeweights_y_test)
print(f"u train:{x_train.shape}, v train:{y_train.shape}, truth train:{truth_train.shape}")
print(f"u test:{x_test.shape}, v test:{y_test.shape}, truth test:{truth_test.shape}")


######################################################################
# Model
######################################################################
kx_max, ky_max = args.kx_max, args.ky_max
ndims = 2
train_sp_L = args.train_sp_L
Lx, Ly = args.Lx, args.Ly
modes = compute_Fourier_modes(ndims, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = ExtGNOBNO(ndims, modes, nmeasures=1,
                  layers=[128, 128, 128, 128, 128],
                  radius=radius,
                  fc_dim=128,
                  in_dim_x=in_dim_x,
                  in_dim_y=4,
                  out_dim=1,
                  train_sp_L=train_sp_L,
                  act="gelu").to(device)


######################################################################
# Training
######################################################################
epochs = 500
base_lr = 0.001
weight_decay = 1.0e-4
batch_size = 20
lr_ratio = args.lr_ratio
normalization_x, normalization_y, normalization_out = args.normalization_x, args.normalization_y, args.normalization_out
normalization_dim_x, normalization_dim_y, normalization_dim_out = [], [], []
non_normalized_dim_x, non_normalized_dim_y, non_normalized_dim_out = 0, 0, 0

checkpoint_path = args.checkpoint_path
save_model_ID = args.save_model_ID

config = {"train": {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay,
                    "epochs": epochs, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_out": normalization_out,
                    "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, "normalization_dim_out": normalization_dim_out,
                    "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y, "non_normalized_dim_out": non_normalized_dim_out,
                    "device": device},
          "test": {"m_tests": m_tests, "shape_types": shape_types_test}}


BNO_train(model,
          x_train, y_train, truth_train, aux_train,
          x_test, y_test, truth_test, aux_test,
          config, checkpoint_path, save_model_ID)

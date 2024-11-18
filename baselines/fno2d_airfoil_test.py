import torch
import sys
import numpy as np
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
np.random.seed(0)
torch.manual_seed(0)

from baselines.fno import FNO2d, FNO_train
from myutils import init_airfoil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "data": {
        "type": "normal",
        "L": 1,
        "downsample_ratio": 1,
        "n_train": 1000,
        "n_test": 200,
        "should_flatten": False,
        "should_compute_bases": False,
    },
    "train": {
        "device": device,
        "base_lr": 0.001,
        "weight_decay": 1.0e-4,
        "epochs": 500,
        "scheduler": "OneCycleLR",
        "batch_size": 20,
        "normalization_x": True,
        "normalization_y": True,
        "normalization_dim": [],
    },
}

data, _, _ = init_airfoil(config)

kx_max, ky_max = 32, 16
###################################
# construct model and train
###################################
model = FNO2d(
    modes1=[kx_max, kx_max, kx_max, kx_max],
    modes2=[ky_max, ky_max, ky_max, ky_max],
    fc_dim=128,
    layers=[128, 128, 128, 128, 128],
    in_dim=2 + 2,
    out_dim=1,
    act="gelu",
    pad_ratio=0.05,
).to(device)


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(data, config, model)

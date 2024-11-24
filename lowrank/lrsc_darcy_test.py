import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt


os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

np.random.seed(0)
torch.manual_seed(0)

from lowrank.lrno_scale import LrGkNO
from myutils import model_train, init_darcy2d

with open("config_lrsc.yml", "r") as f:
    config = dict(yaml.full_load(f)["darcy2d"])
data, bases, _ = init_darcy2d(config)

# id_base = bases["id_out"][0]
# fig, ax = plt.subplots(ncols=4)
# for i in range(4):
#     sample = id_base[..., 2 * i].cpu()
#     ax[i].imshow(sample, cmap="viridis", interpolation="nearest", extent=(0, 1, 0, 1))
# plt.show()

model = LrGkNO(bases, **config["model"]).to("cuda")
model_train(data, config, model)

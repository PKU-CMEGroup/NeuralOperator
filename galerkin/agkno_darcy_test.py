import os
import sys
import yaml
import torch
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
np.random.seed(0)
torch.manual_seed(0)


from Galerkin.agkno import AdjGkNO
from myutils import model_train, init_darcy2d

with open("config_adjust.yml", "r") as f:
    config = dict(yaml.full_load(f)["darcy2d"])
data, bases, _ = init_darcy2d(config)

model = AdjGkNO(bases, **config["model"]).to("cuda")
model_train(data, config, model)
import os
import sys
import yaml
import torch
import numpy as np


os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

np.random.seed(0)
torch.manual_seed(0)

from lowrank.lrno_noscale import LrGkNO
from myutils import model_train, init_airfoil

with open("config_lr.yml", "r") as f:
    config = dict(yaml.full_load(f)["airfoil"])
data, bases, _ = init_airfoil(config)

model = LrGkNO(bases, **config["model"]).to("cuda")
model_train(data, config, model)

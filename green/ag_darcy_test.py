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

from green.attngreen import AttnGkNO
from myutils import model_train, init_darcy2d

with open("config_ag.yml", "r") as f:
    config = dict(yaml.full_load(f)["darcy2d"])
data, bases, _ = init_darcy2d(config)


model = AttnGkNO(bases, **config["model"]).to("cuda")
model_train(data, config, model)

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


from attention.gat import GAT
from myutils import graph_train, init_airfoil_graph

with open("config_graph.yml", "r") as f:
    config = dict(yaml.full_load(f)["airfoil"])
train_loader, test_loader, y_normalizer, G0 = init_airfoil_graph(config)

model = GAT(**config["model"]).to("cuda")
graph_train(train_loader, test_loader, y_normalizer, config, model)
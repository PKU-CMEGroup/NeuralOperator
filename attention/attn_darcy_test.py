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

from attention.attnno import AttnNO
from myutils import model_train, init_darcy2d, indices_neighbor2d


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("config.yml", "r") as f:
    config = dict(yaml.full_load(f)["darcy2d"])
data, _, grid = init_darcy2d(config)

n1 = grid["n1"]
n2 = grid["n2"]
k1 = 2
k2 = 2
indices = indices_neighbor2d(n1, n2, k1, k2, should_flatten=True)
print(f"indices size {indices.shape}")

model = AttnNO(indices, **config["model"]).to("cuda")
model_train(data, config, model)

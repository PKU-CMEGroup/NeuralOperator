import sys
import os
import yaml
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

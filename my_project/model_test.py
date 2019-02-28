import torch
from config import Config
import numpy as np
import torch.nn as nn
from model.mlp import MLP

x = torch.LongTensor(np.ones(shape=(128, 8)))
model = MLP(Config)
print(model(x).size())
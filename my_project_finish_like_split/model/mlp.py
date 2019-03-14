from .base_model import BaseModel
import torch
from config import Config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module, BaseModel):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.model_config = config["model_config"]["MLP_config"]
        self.data_config = config["data_config"]
        model_config = config["model_config"]["MLP_config"]
        data_config = config["data_config"]
        k = model_config["k"]
        num_feature = len(data_config)
        layers = model_config["layers"]
        self.emb_layers = nn.ModuleList([nn.Embedding(data_config[key], model_config["k"]) for key in data_config])
        self.linear1 = nn.Linear(in_features=num_feature*k, out_features=layers[0])
        self.linear_layers = [nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))]

    def forward(self, x):
        first_layer = []
        for i in range(x.size()[1]):
            first_layer.append(self.emb_layers[i](x[:, i]))
        x = torch.cat(first_layer, 1)
        x = self.linear1(x)
        for layer in self.linear_layers:
            x = layer(x)
        x = F.sigmoid(x)
        return x[:, 0], x[:, 1]



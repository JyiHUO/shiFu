import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torch import nn

class FM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def forward(self, x):
        pass
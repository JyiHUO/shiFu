import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class BaseModel:
    def __init__(self):
        self.path_config = None
        model_config = None
        data_config = None

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        self.load_state_dict(torch.load(self.path_config["normal_config"]["pretrain_model_dir"]))
        self.eval()






